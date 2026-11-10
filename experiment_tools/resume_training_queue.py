import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback

ROOT=Path('/root/ProMamba_Project')
OLD=ROOT/'experiments/chapter6_priority_20260917'
OUT=ROOT/'experiments/chapter6_resume_20260920'
PY='/root/miniconda3/envs/airnet/bin/python'
SESSION='chapter6_resume_20260920'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):(OUT/name).write_text(json.dumps(v,indent=2))
def event(kind,**kw):
    r=dict(time=now(),event=kind,**kw)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)
def original():
    spec=importlib.util.spec_from_file_location('original_worker',OLD/'worker.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def idle():
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
    assert not active,'GPU occupied: '+active
def prepare():
    idle();m=original();m.verify()
    assert not (OUT/'launch_report.json').exists()
    import torch
    run=ROOT/'runs/chapter6_priority_20260917/F1_FDMB_conv'
    c=torch.load(str(run/'latest.pth'),map_location='cpu')
    assert c['epoch']==44
    for k in ['optimizer_state_dict','scheduler_state_dict','scaler_state_dict','ema_state_dict','torch_rng_state','python_rng_state','numpy_rng_state','data_generator_state']:
        assert k in c,k
    for section in ['model_state_dict','ema_state_dict']:
        assert all(torch.isfinite(t).all() for t in c[section].values())
    assert c['model_config']['bottleneck_type']=='conv'
    assert not (ROOT/'runs/chapter6_priority_20260917/L0_eta0').exists()
    backup=OUT/'backup_FDMB_epoch44';backup.mkdir()
    hashes={}
    for name in ['latest.pth','best.pth','metrics.csv','config.json']:
        shutil.copy2(run/name,backup/name)
        assert sha(run/name)==sha(backup/name)
        hashes[name]=sha(backup/name)
    plfb=json.loads((ROOT/'evaluations/chapter6_priority_20260917/P1_PLFB_local/metrics.json').read_text())
    assert sum(r['count'] for r in plfb['conditions'].values())==804
    jobs=[]
    for j in m.PLAN['jobs']:
        if j['name']=='P1_PLFB_local':continue
        j=dict(j);j['train_command']=list(j['train_command'])
        if j['name']=='F1_FDMB_conv':j['train_command']+=['--resume',str(run/'latest.pth')]
        jobs.append(j)
    save('launch_report.json',dict(created_at=now(),jobs=jobs,backup_sha256=hashes,session=SESSION,
        source_worker_sha256=sha(OLD/'worker.py'),resume_epoch=45,skip_completed=['P1_PLFB_local','sensitivity','initial_efficiency']))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    command=shlex.join([PY,'-u','-B',str(OUT/'resume.py'),'worker'])+' > '+shlex.quote(str(OUT/'queue.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',SESSION,'-c',str(OUT),command],env=env,check=True)
    print(json.dumps({'launched':SESSION,'resume_epoch':45,'backup_verified':True}))
def worker():
    with (OLD/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle();m=original();m.verify()
        report=json.loads((OUT/'launch_report.json').read_text())
        assert sha(OLD/'worker.py')==report['source_worker_sha256']
        import torch
        for j in report['jobs']:
            m.verify();name=j['name'];run=ROOT/'runs/chapter6_priority_20260917'/name
            for phase,command in [('training',j['train_command']),('testing',j['test_command']),
                ('efficiency',[PY,'-u','-B',str(OLD/'worker.py'),'efficiency',name,str(run/'best.pth')])]:
                if phase=='testing':
                    c=torch.load(str(run/'latest.pth'),map_location='cpu');assert c['epoch']==100;del c
                    c=torch.load(str(run/'best.pth'),map_location='cpu')
                    assert all(torch.isfinite(t).all() for t in c['ema_state_dict'].values())
                    event('checkpoints_verified',name=name,best_epoch=c['epoch']);del c
                with (OUT/(name+'.'+phase+'.log')).open('x') as log:
                    p=subprocess.Popen(command,cwd=str(m.CODE),stdout=log,stderr=subprocess.STDOUT)
                    event('process_started',name=name,phase=phase,pid=p.pid)
                    code=p.wait()
                event('process_exited',name=name,phase=phase,exit_code=code)
                if code:raise RuntimeError(name+' '+phase+' failed')
                if phase=='testing':
                    r=json.loads((ROOT/'evaluations/chapter6_priority_20260917'/name/'metrics.json').read_text())
                    assert sum(x['count'] for x in r['conditions'].values())==804
                    event('test_verified',name=name,avg5_psnr=r['avg5_psnr'],avg5_ssim=r['avg5_ssim'])
        event('queue_completed');save('exit_status.json',dict(exit_code=0,finished_at=now()))
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        try:worker()
        except Exception:
            save('exit_status.json',dict(exit_code=1,finished_at=now(),traceback=traceback.format_exc()));raise
