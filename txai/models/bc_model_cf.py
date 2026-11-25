import torch
from torch import nn
import torch.nn.functional as F

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.functional import transform_to_attn_mask
from txai.models.mask_generators.maskgen import MaskGenerator
from tint.models import MLP, RNN

from txai.utils.predictors.loss import GSATLoss, ConnectLoss
from txai.utils.predictors.loss_smoother_stats import *
from txai.utils.functional import js_divergence, stratified_sample
from txai.models.encoders.simple import CNN, LSTM

transformer_default_args = {
    'enc_dropout': None,
    'nhead': 1,
    'trans_dim_feedforward': 72,
    'trans_dropout': 0.25,
    'nlayers': 1,
    'aggreg': 'mean',
    'MAX': 10000,
    'static': False,
    'd_static': 0,
    'd_pe': 16,
    'norm_embedding': True,
}


all_default_opt_kwargs = {
    'lr': 0.0001,
    'weight_decay': 0.01,
} 


from dataclasses import dataclass, field
@dataclass
class AblationParameters:
    equal_g_gt: bool = field(default = False)
    hard_concept_matching: bool = field(default = True)
    use_loss_on_concept_sims: bool = field(default = False)
    use_concept_corr_loss: bool = field(default = False)
    g_pret_equals_g: bool = field(default = False)
    label_based_on_mask: bool = field(default = False)
    use_ste: bool = field(default = True)
    # Prototypes:
    ptype_assimilation: bool = field(default = False)
    side_assimilation: bool = field(default = False)
    archtype: str = field(default = 'transformer')
    cnn_dim: int = field(default = 128)

default_abl = AblationParameters() # Class based on only default params

default_loss_weights = {
    'gsat': 1.0,
    'connect': 1.0,
}

class DistributionParams(nn.Module):  
    def __init__(self, input_dim, out_dim, hidden_dim=32):  
        super().__init__()  
        self.fc_mean = MLP([input_dim, hidden_dim, out_dim], activations='elu', dropout=0.0)
        self.fc_logvar = MLP([input_dim, hidden_dim, out_dim], activations='elu', dropout=0.0)
        # self.fc_mean = nn.Linear(input_dim, hidden_dim)  
        # self.fc_logvar = nn.Linear(input_dim, hidden_dim) 
    def forward(self, x):  
        mean = self.fc_mean(x)  
        logvar = self.fc_logvar(x)  
        return mean, logvar  

class TimeXModel(nn.Module):
    '''
    Model has full options through config
        - Use for ablations - configuration supported through config load
    '''
    def __init__(self,
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            n_prototypes,
            gsat_r, 
            transformer_args = transformer_default_args,
            ablation_parameters = default_abl,
            loss_weight_dict = default_loss_weights,
            tau = 1.0,
            mlp_dim = 32,
            masktoken_stats = None,
        ):
        super(TimeXModel, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = transformer_default_args['d_pe']
        self.n_classes = n_classes
        self.transformer_args = transformer_args
        self.n_prototypes = n_prototypes
        self.gsat_r = gsat_r
        self.tau = tau
        self.masktoken_stats = masktoken_stats

        self.ablation_parameters = ablation_parameters
        self.loss_weight_dict = loss_weight_dict
        
        # Holds main encoder:
        if self.ablation_parameters.archtype == 'transformer':
            self.encoder_main = TransformerMVTS(
                d_inp = d_inp,  # Dimension of input from samples (must be constant)
                max_len = max_len, # Max length of any sample to be fed into model
                n_classes = self.n_classes, # Number of classes for classification head
                **self.transformer_args
            )
            self.d_z = (self.d_inp + self.d_pe)
        elif self.ablation_parameters.archtype == 'cnn':
            self.encoder_main = CNN(
                d_inp = d_inp,
                n_classes = self.n_classes,
                dim = self.ablation_parameters.cnn_dim # Abuse of notation, but just for experiment purposes
            )
            self.d_z = self.ablation_parameters.cnn_dim
        elif self.ablation_parameters.archtype == 'lstm':
            self.encoder_main = LSTM(
                d_inp = d_inp,
                n_classes = self.n_classes,
            )
            self.d_z = 128

        self.encoder_pret = TransformerMVTS(
            d_inp = d_inp,  # Dimension of input from samples (must be constant)
            max_len = max_len, # Max length of any sample to be fed into model
            n_classes = self.n_classes, # Number of classes for classification head
            **self.transformer_args # TODO: change to a different parameter later - leave simple for now
        )

        # For decoder, first value [0] is actual value, [1] is mask value (predicted logit)
        self.mask_generator = MaskGenerator(d_z = (self.d_inp + self.d_pe), d_pe = self.d_pe, max_len = max_len, tau = self.tau, 
            use_ste = self.ablation_parameters.use_ste)

        self.mask_connection_src = MLP([2, mlp_dim, 1], activations='elu', dropout=0.0)
        self.perturbation_generator = MLP([2, mlp_dim, 1], activations='elu', dropout=0.0)
        self.noise_generator = MLP([3, mlp_dim, 1], activations='elu', dropout=0.0)
        self.mask_src_distribution = DistributionParams(self.d_inp*max_len, self.d_inp*max_len)
        self.src_distribution = DistributionParams(self.d_inp*max_len, self.d_inp*max_len)

        # Setup loss functions:
        self.gsat_loss_fn = GSATLoss(r = self.gsat_r)
        self.connected_loss = ConnectLoss()

        self.set_config()

    def forward(self, src, times, captum_input = False, reference_x = None, target_label = None):
        # TODO: return early from function when in eval
        
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)
            if reference_x is not None:
                reference_x = reference_x.transpose(0, 1)

        if self.ablation_parameters.archtype == 'transformer':

            pred_regular, z_main, z_seq_main = self.encoder_main(src, times, captum_input = False, get_agg_embed = True)
        else:

            pred_regular, z_main = self.encoder_main(src, times, captum_input = False, get_embedding = True)

        if not self.ablation_parameters.g_pret_equals_g:
            z_seq = self.encoder_pret.embed(src, times, captum_input = False, aggregate = False)

        # Generate smooth_src: # TODO: expand to lists
        smooth_src_list, mask_in_list, ste_mask_list, = [], [], []

        if self.ablation_parameters.g_pret_equals_g:
            mask_in, ste_mask = self.mask_generator(src, times, z_seq=z_seq_main, training=self.training)
        else:
            mask_in, ste_mask = self.mask_generator(src, times, z_seq=z_seq, training=self.training)

        # exp_src, reference_src = self.multivariate_mask(src, ste_mask)

        # if self.ablation_parameters.archtype == 'transformer':
        #     pred_mask, z_mask, z_seq_mask = self.encoder_main(exp_src, times, get_agg_embed = True)
        # else:
        #     pred_mask, z_mask = self.encoder_main(exp_src, times, get_embedding = True)

        # if self.ablation_parameters.label_based_on_mask:
        #     pred_mask = self.z_e_predictor(z_mask) # Make prediction on masked input
        ste_mask_rs = ste_mask.transpose(0, 1)
        if len(ste_mask_rs.shape) == 2:
            ste_mask_rs = ste_mask_rs.unsqueeze(-1)

        e_raw = self.perturbation_generator(torch.stack([src, ste_mask_rs], dim=-1)).squeeze(-1)
        e_cf = e_raw * ste_mask_rs

        if reference_x is None:
            reference_x = src
        
        # noise = self._get_baseline(B = src.shape[1])
        if self.training:
            noise = self.noise_generator(torch.stack([src, ste_mask_rs, reference_x], dim=-1)).squeeze(-1)
            noise = noise - noise.mean(dim=-1, keepdim=True)
        else:
            noise = torch.zeros_like(src)
        x_prime_cf = src + e_cf + noise

        if self.ablation_parameters.archtype == 'transformer':
            pred_cf, _, _ = self.encoder_main(x_prime_cf, times, get_agg_embed = True)
        else:
            pred_cf, _ = self.encoder_main(x_prime_cf, times, get_embedding = True)

        ref_distr = self.src_distribution(reference_x.transpose(0, 1).reshape(-1, self.max_len * self.d_inp))
        cf_distr = self.mask_src_distribution(
            x_prime_cf.transpose(0, 1).reshape(-1, self.max_len * self.d_inp)
        )

        total_out_dict = {
            'pred': pred_regular, # Prediction on regular embedding (prediction branch)
            # 'pred_mask': pred_mask, # Prediction on masked embedding
            'pred_cf': pred_cf, # Prediction on counterfactual candidate
            'mask_logits': mask_in, # Mask logits, i.e. before reparameterization + ste
            'ste_mask': ste_mask,
            'smooth_src': src,                                  
            'all_z': (ref_distr, cf_distr),             # Keep in src domain for KL(ref || cf)
            'reference_z': (src, x_prime_cf),           # Backward compatible key expected by eval
            # 'reference_src': reference_src,
            # 'exp_src': exp_src,
            'reference_x': reference_x,
            'noise': noise,
            'E_raw': e_raw,
            'E_cf': e_cf,
            'X_prime_cf': x_prime_cf,
            'X_cf': src + e_cf,
            'target_label': target_label,
            # 'vis': (z_main, z_mask),
        }

        return total_out_dict

    def get_saliency_explanation(self, src, times, captum_input = False):
        '''
        Retrieves only saliency explanation (not concepts)
            - More efficient than calling forward due to less module calls
        '''

        if self.ablation_parameters.g_pret_equals_g:
            z_seq = self.encoder_main.embed(src, times, captum_input = False, aggregate = False)
        else:
            z_seq = self.encoder_pret.embed(src, times, captum_input = False, aggregate = False)

        mask_in, ste_mask = self.mask_generator(src, times, z_seq=z_seq)

        out_dict = {
            'smooth_src': src,
            'mask_in': mask_in.transpose(0,1),
            'ste_mask': ste_mask,
        }

        return out_dict
    
    def multivariate_mask(self, src, ste_mask):
        # First apply mask directly on input:
        baseline = self._get_baseline(B = src.shape[1])
        ste_mask_rs = ste_mask.transpose(0,1)
        if len(ste_mask_rs.shape) == 2:
            ste_mask_rs = ste_mask_rs.unsqueeze(-1)


        src_masked_ref = src * ste_mask_rs + (1 - ste_mask_rs) * baseline#self.baseline_net(src)#baseline
        
        # src_masked = self.mask_connection_src(torch.stack([src * ste_mask_rs, (1 - ste_mask_rs) * baseline], dim=-1)).squeeze(-1)
        src_masked = self.mask_connection_src(torch.stack([src, ste_mask_rs], dim=-1)).squeeze(-1)

        return src_masked, src_masked_ref
    
    def _get_baseline(self, B):
        mu, std = self.masktoken_stats
        samp = torch.stack([torch.normal(mean = mu, std = std) for _ in range(B)], dim = 1)
        return samp

    def compute_loss(self, output_dict):
        mask_loss = self.loss_weight_dict['gsat'] * self.gsat_loss_fn(output_dict['mask_logits']) + self.loss_weight_dict['connect'] * self.connected_loss(output_dict['mask_logits'])
        return mask_loss

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)


    def set_config(self):
        self.config = {
            'd_inp': self.d_inp,
            'max_len': self.max_len,
            'n_classes': self.n_classes,
            'n_prototypes': self.n_prototypes,
            'gsat_r': self.gsat_r,
            'transformer_args': self.transformer_args,
            'ablation_parameters': self.ablation_parameters,
            'tau': self.tau,
            'masktoken_stats': self.masktoken_stats,
        }