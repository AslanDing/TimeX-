"""Minimal job queue: python run_queue.py jobs.txt --gpus 0,4 --per_gpu 4

jobs.txt: one job per line, "<name>\t<command>". Each job's stdout/stderr goes to logs/<name>.log.
A job is skipped if logs/<name>.done exists. Lines starting with # are ignored.
"""
import argparse, os, subprocess, time, sys

ap = argparse.ArgumentParser()
ap.add_argument('jobs')
ap.add_argument('--gpus', default='0')
ap.add_argument('--per_gpu', type=int, default=4)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
logdir = os.path.join(here, 'logs')
os.makedirs(logdir, exist_ok=True)
jobs = []
for line in open(a.jobs):
    line = line.rstrip('\n')
    if not line.strip() or line.startswith('#'):
        continue
    name, cmd = line.split('\t', 1)
    if os.path.exists(os.path.join(logdir, name + '.done')):
        continue
    jobs.append((name, cmd))
gpus = a.gpus.split(',')
slots = {g: [] for g in gpus}
print(time.strftime('%H:%M:%S'), f'{len(jobs)} jobs', flush=True)
while jobs or any(slots.values()):
    for g in gpus:
        alive = []
        for name, p in slots[g]:
            if p.poll() is None:
                alive.append((name, p))
            else:
                ok = p.returncode == 0
                if ok:
                    open(os.path.join(logdir, name + '.done'), 'w').close()
                print(time.strftime('%H:%M:%S'), 'finished', name, 'rc', p.returncode, flush=True)
        slots[g] = alive
        while jobs and len(slots[g]) < a.per_gpu:
            name, cmd = jobs.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED='1')
            f = open(os.path.join(logdir, name + '.log'), 'w')
            p = subprocess.Popen(cmd, shell=True, cwd=here, env=env, stdout=f, stderr=subprocess.STDOUT)
            slots[g].append((name, p))
            print(time.strftime('%H:%M:%S'), 'started', name, 'on gpu', g, flush=True)
    time.sleep(5)
print(time.strftime('%H:%M:%S'), 'all done', flush=True)
