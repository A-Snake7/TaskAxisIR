"""Fail-stop priority experiments; isolated source, fixed data and budgets."""
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

OUT=Path('/root/ProMamba_Project/experiments/chapter6_priority_20260917')
ROOT=OUT.parents[1]
CODE=OUT/'code'
PREV=ROOT/'experiments/chapter6_supplement_20260917'
PY='/root/miniconda3/envs/airnet/bin/python'
PLAN=json.loads((OUT/'plan.json').read_text())
sys.path.insert(0,str(CODE))

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2))
def event(kind,**extra):
    r=dict(time=now(),event=kind,**extra)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)
def verify():
    for name,digest in PLAN['source_sha256'].items():assert sha(CODE/name)==digest,name
    assert sha(PLAN['baseline']['weights'])==PLAN['baseline']['checkpoint_sha256']
    for k in ['train_manifest','val_manifest']:
        r=PLAN['baseline_data_protocol'][k];assert sha(r['path'])==r['sha256']
def setup():
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    return torch
def model_for(name,weights=None):
    from utils.checkpoint import load_model_for_inference
    from net.promamba import ProMamba
    if weights:return load_model_for_inference(weights,'cuda')[0]
    config=dict(PLAN['baseline']['model_config'])
    if name=='P1_PLFB_local':config['use_frequency_branch']=False
    elif name=='F1_FDMB_conv':config['bottleneck_type']='conv'
    elif name=='L0_eta0':config['local_condition_scale']=0.
    else:return load_model_for_inference(PLAN['baseline']['weights'],'cuda')[0]
    return ProMamba(**config).cuda()

def operation_counts(model,x):
    """Count dense/conv MACs, including fused Mamba projections analytically.

    Deliberately not a total FLOP count: scan recurrence arithmetic and exponentials
    are separate, and activation/normalization/pooling/gating costs are excluded.
    """
    import torch
    from torch import nn
    names={id(m):n for n,m in model.named_modules()}
    mambas={n:m for n,m in model.named_modules() if type(m).__name__=='Mamba'}
    totals=dict(dense_conv_macs=0,scan_recurrence_arithmetic_flops_estimate=0,scan_exponentials=0)
    handles=[]
    def count(module,args,out):
        if isinstance(module,nn.Conv2d):
            totals['dense_conv_macs']+=out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module,nn.Conv1d):
            totals['dense_conv_macs']+=out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
        elif isinstance(module,nn.Linear):totals['dense_conv_macs']+=out.numel()*module.in_features
    def mamba_count(m,args,out):
        b,l,d=args[0].shape;inner=m.d_inner;n=m.d_state;r=m.dt_rank
        totals['dense_conv_macs']+=b*l*(d*2*inner+inner*m.d_conv+inner*(r+2*n)+r*inner+inner*d)
        # delta*A, delta*B*u, state update, C readout, skip D*u; no silu/softplus.
        totals['scan_recurrence_arithmetic_flops_estimate']+=b*l*inner*(7*n+1)
        totals['scan_exponentials']+=b*l*inner*n
    for name,m in model.named_modules():
        if name in mambas:handles.append(m.register_forward_hook(mamba_count))
        elif not any(name.startswith(n+'.') for n in mambas) and isinstance(m,(nn.Conv2d,nn.Conv1d,nn.Linear)):
            handles.append(m.register_forward_hook(count))
    with torch.inference_mode():model(x,None)
    for h in handles:h.remove()
    return dict(**totals,status='analytic_major_operations_not_total_flops',
        convention='One multiply-accumulate counts as one MAC. Mamba projections+depthwise convolution counted once even when fused. Scan recurrence ~B*L*d_inner*(7*d_state+1) scalar arithmetic FLOPs; exp counted separately.',
        excludes=['bias adds','normalization','activations including softplus and SiLU','pooling','interpolation','routing and fusion elementwise operations','kernel implementation overhead'])

def benchmark(fn,torch,warmup=10,repeats=30):
    with torch.inference_mode():
        for _ in range(warmup):fn()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        base=torch.cuda.memory_allocated();times=[]
        for _ in range(repeats):
            torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        peak=torch.cuda.max_memory_allocated()
    return dict(warmup=warmup,repeats=repeats,latency_ms_mean=statistics.mean(times),
        latency_ms_median=statistics.median(times),latency_ms_p95=sorted(times)[int(.95*(len(times)-1))],
        peak_allocated_mib=peak/2**20,incremental_peak_mib=(peak-base)/2**20,
        method='Synchronized wall clock; includes Python routing and GPU execution; excludes input transfer and file IO')

def efficiency(name,weights=None):
    torch=setup();verify();torch.manual_seed(20260917)
    model=model_for(name,weights).eval()
    x=torch.rand(1,3,256,256,device='cuda')
    result=dict(name=name,weights=weights or ('baseline EMA' if name=='Full' else 'random initialization architecture-only preliminary'),
        parameters=sum(p.numel() for p in model.parameters()),batch_size=1,input_shape=list(x.shape),precision='fp32',
        tf32_matmul=False,tf32_cudnn=True,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda=torch.version.cuda,
        operations=operation_counts(model,x),full_image=benchmark(lambda:model(x,None),torch))
    if name=='Full':
        from utils.chapter6_inference import tiled_inference
        x=torch.rand(1,3,512,512,device='cuda')
        result['routing_512_input_256_tile_32_overlap']={}
        for mode in ['global_shared','independent_tiles','shared_aggregate']:
            fn=lambda:tiled_inference(model,x,amp_enabled=False,independent_tile_routes=(mode=='independent_tiles'),shared_global_route=(mode=='global_shared'))
            result['routing_512_input_256_tile_32_overlap'][mode]=benchmark(fn,torch)
    save('efficiency_'+name+('_trained' if weights else '')+'.json',result)
    event('efficiency_completed',name=name,parameters=result['parameters'])

def preflight():
    torch=setup();verify()
    from utils.chapter6_inference import tiled_inference
    from scripts import evaluate_aio3_promptir as ev
    ev.self_test()
    manifest=ROOT/'.testing_launches/taskaxisir_blind_v3_scratch_best84_aio3_fp32_full_20260911_s42/test_file_manifest.json'
    for r in json.loads(manifest.read_text()):assert sha(r['path'])==r['sha256']
    checks=[]
    for name in ['P1_PLFB_local','F1_FDMB_conv','L0_eta0']:
        torch.manual_seed(42);m=model_for(name).train()
        if name=='P1_PLFB_local':assert all(not b.use_frequency_branch for b in m.modules() if type(b).__name__=='PromptLocalFrequencyBlock')
        if name=='F1_FDMB_conv':assert not any(type(b).__name__=='Mamba' for b in m.modules())
        if name=='L0_eta0':assert m.hierarchical.local_condition_scale==0
        opt=torch.optim.AdamW(m.parameters(),lr=2e-4)
        for step in range(2):
            x=torch.rand(1,3,64,64,device='cuda');y=torch.rand_like(x)
            out,aux=m(x,None,return_aux=True)
            loss=(out-y).square().mean()+.1*torch.nn.functional.cross_entropy(aux['task_logits'],torch.tensor([0],device='cuda'))
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            opt.step()
        checks.append(dict(name=name,two_optimizer_steps_passed=True,parameters=sum(p.numel() for p in m.parameters())))
        del m,opt,out,aux,loss;torch.cuda.empty_cache()
    m=model_for('Full').eval()
    path=ROOT/'data/test/derain/input/rain-001.png'
    x=ev.uint8_to_tensor(ev.load_rgb_uint8(path,16),torch.device('cuda'))
    with torch.inference_mode():
        context=m.predict_task_context(x)
        global_probs=context['global_task_probs']
        calls=[]
        def hook(module,args,kwargs):calls.append(kwargs['routing_probs_override'].detach().clone())
        handle=m.register_forward_pre_hook(hook,with_kwargs=True)
        global_out=tiled_inference(m,x,amp_enabled=False,shared_global_route=True)
        handle.remove()
        assert calls and all(torch.allclose(p,global_probs,atol=1e-7,rtol=1e-6) for p in calls)
        shared=tiled_inference(m,x,amp_enabled=False)
        index=ev.build_target_index(ROOT/'data/test/derain/target')
        target=ev.load_rgb_uint8(ev.find_target(path,index,'derain'),16).astype('float32')/255
        psnr,ssim=ev.compute_metrics(ev.tensor_to_rgb_float(shared),target)
        previous=next(r for r in csv.DictReader((PREV/'aggregation_per_image.csv').open()) if r['mode']=='shared_aggregate' and r['image']==path.name)
        assert abs(psnr-float(previous['psnr']))<1e-5
        assert abs(ssim-float(previous['ssim']))<1e-6
    checks.append(dict(global_route_reused_all_tiles=True,baseline_psnr_reproduced=psnr))
    save('preflight.json',dict(status='passed',checks=checks,dataset_files_verified=1328))
    event('preflight_passed')

def inference():
    torch=setup();verify()
    import importlib.util
    import numpy as np
    from utils.chapter6_inference import tiled_inference
    from scripts import evaluate_aio3_promptir as ev
    spec=importlib.util.spec_from_file_location('previous_eval',PREV/'evaluate.py');prev=importlib.util.module_from_spec(spec);spec.loader.exec_module(prev)
    model=model_for('Full').eval();results={}
    for tile,overlap in PLAN['sensitivity_settings']:
        tag='tile%d_overlap%d'%(tile,overlap)
        rows=[]
        reused=[]
        if (tile,overlap)==(256,32):
            for r in csv.DictReader((PREV/'aggregation_per_image.csv').open()):
                r['psnr']=float(r['psnr']);r['ssim']=float(r['ssim']);r['tile_count']=int(r['tile_count']);rows.append(r)
            reused=['shared_aggregate','independent_tiles']
        with (OUT/(tag+'_per_image.jsonl')).open('x') as log:
            for i,(condition,name,degraded,clean) in enumerate(prev.inputs(),1):
                x=ev.uint8_to_tensor(degraded,torch.device('cuda'))
                for mode in ['global_shared','independent_tiles','shared_aggregate']:
                    if mode in reused:continue
                    with torch.inference_mode():
                        out,aux=tiled_inference(model,x,tile_size=tile,overlap=overlap,amp_enabled=False,return_routing=True,
                            independent_tile_routes=(mode=='independent_tiles'),shared_global_route=(mode=='global_shared'))
                    psnr,ssim=ev.compute_metrics(ev.tensor_to_rgb_float(out),clean.astype(np.float32)/255)
                    assert np.isfinite(psnr) and np.isfinite(ssim)
                    row=dict(mode=mode,condition=condition,image=name,psnr=psnr,ssim=ssim,tile_count=aux['tile_count'])
                    rows.append(row);log.write(json.dumps(row)+'\n')
                if i%100==0:event('sensitivity_progress',setting=tag,images=i,total=804)
                log.flush()
        assert len(rows)==3*804
        setting={}
        for mode in ['global_shared','independent_tiles','shared_aggregate']:
            setting[mode]={}
            for c in prev.CONDITIONS:
                r=[r for r in rows if r['mode']==mode and r['condition']==c]
                assert len(r)==(100 if c=='derain' else 500 if c=='dehaze' else 68)
                setting[mode][c]={m:float(np.mean([x[m] for x in r])) for m in ['psnr','ssim']}
            setting[mode]['avg5']={m:float(np.mean([setting[mode][c][m] for c in prev.CONDITIONS])) for m in ['psnr','ssim']}
        with (OUT/(tag+'_per_image.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        results[tag]=setting
        save('sensitivity_results.json',dict(status='partial' if len(results)<5 else 'completed',settings=results,
            checkpoint_sha256=PLAN['baseline']['checkpoint_sha256'],precision='fp32',tf32_matmul=False,tf32_cudnn=True,conditions=prev.CONDITIONS,
            note='Fixed full-image global condition. Both non-global modes execute identical routing-only tile pass. Reused 256/32 pair from prior verified evaluation.'))
        event('sensitivity_setting_completed',setting=tag,avg5={m:s['avg5'] for m,s in setting.items()})
    event('inference_completed')

def run(command,logname):
    with (OUT/logname).open('x') as f:
        p=subprocess.Popen(command,cwd=str(CODE),stdout=f,stderr=subprocess.STDOUT)
        event('process_started',pid=p.pid,log=logname,command=command)
        code=p.wait()
    event('process_exited',log=logname,exit_code=code)
    if code:raise RuntimeError('Failed: '+logname)

def queue():
    with (OUT/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify()
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
        assert not active,'GPU occupied: '+active
        assert not (OUT/'preflight.json').exists(),'Inspect existing work before restarting'
        run([PY,'-u','-B',str(OUT/'worker.py'),'preflight'],'preflight.log')
        for name in ['Full','P1_PLFB_local','F1_FDMB_conv','L0_eta0']:
            run([PY,'-u','-B',str(OUT/'worker.py'),'efficiency',name],'efficiency_'+name+'.log')
        run([PY,'-u','-B',str(OUT/'worker.py'),'inference'],'inference.log')
        for job in PLAN['jobs']:
            verify();name=job['name'];runpath=ROOT/'runs/chapter6_priority_20260917'/name
            assert not runpath.exists(),'Refusing overwrite: '+str(runpath)
            event('training_started',name=name)
            run(job['train_command'],name+'.train.log')
            # Verify final epoch and finite best EMA before formal test.
            import torch
            latest=torch.load(str(runpath/'latest.pth'),map_location='cpu')
            assert latest['epoch']==100
            del latest
            best=torch.load(str(runpath/'best.pth'),map_location='cpu')
            assert all(torch.isfinite(t).all() for t in best['ema_state_dict'].values())
            event('training_verified',name=name,best_epoch=best['epoch']);del best
            run(job['test_command'],name+'.test.log')
            result=json.loads((ROOT/'evaluations/chapter6_priority_20260917'/name/'metrics.json').read_text())
            assert sum(v['count'] for v in result['conditions'].values())==804
            event('test_completed',name=name,avg5_psnr=result['avg5_psnr'],avg5_ssim=result['avg5_ssim'])
            run([PY,'-u','-B',str(OUT/'worker.py'),'efficiency',name,str(runpath/'best.pth')],name+'.efficiency_trained.log')
        event('queue_completed')
        save('exit_status.json',dict(exit_code=0,finished_at=now()))

if __name__=='__main__':
    mode=sys.argv[1]
    try:
        if mode=='queue':queue()
        elif mode=='preflight':preflight()
        elif mode=='inference':inference()
        elif mode=='efficiency':efficiency(sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
        else:raise ValueError(mode)
    except Exception:
        if mode=='queue':save('exit_status.json',dict(exit_code=1,finished_at=now(),traceback=traceback.format_exc()))
        raise
