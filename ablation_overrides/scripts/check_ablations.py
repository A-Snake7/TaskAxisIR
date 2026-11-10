"""Check route algebra, teacher gradients, checkpoint reload and real CUDA steps."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

CODE = Path(__file__).resolve().parents[1]
ROOT = Path('/root/ProMamba_Project')
sys.path.insert(0, str(CODE))
import train
from net.promamba_v2 import ReliabilityAwareRoutingFusion, HierarchicalProMamba
from utils.losses import blind_v3_routing_loss, RestorationLoss
from utils.checkpoint import load_model_for_inference
from utils.inference import tiled_inference
from dataset import AllInOneDataset


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    out = Path(plan['directory']) / 'checks'
    out.mkdir(exist_ok=False)
    original = load_source('original_v3', ROOT / 'net/promamba_v2.py')
    old_loss = load_source('original_losses', ROOT / 'utils/losses.py')
    torch.manual_seed(42)
    old = original.HierarchicalProMamba(dim=48, stage_blocks=(2,3,4), condition_mode='blind_v3')
    state = old.state_dict()
    for mode in ['learned', 'equal', 'global', 'local']:
        torch.manual_seed(42)
        new = HierarchicalProMamba(dim=48, stage_blocks=(2,3,4), condition_mode='blind_v3', v3_route_mode=mode)
        assert set(state) == set(new.state_dict())
        assert all(torch.equal(v, new.state_dict()[k]) for k,v in state.items()), mode
    del old,new,state
    g = torch.tensor([[3.,0.,-1.],[0.,1.,3.]], requires_grad=True)
    l = torch.tensor([[0.,3.,-1.],[1.,0.,3.]], requires_grad=True)
    gf = torch.randn(2,48,requires_grad=True)
    lf = torch.randn(2,48,requires_grad=True)
    old_gate = original.ReliabilityAwareRoutingFusion(48,3)
    new_gate = ReliabilityAwareRoutingFusion(48,3)
    new_gate.load_state_dict(old_gate.state_dict())
    a,b = old_gate(g,l,gf,lf),new_gate(g,l,gf,lf)
    for k in a: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    for mode,alpha in [('equal',.5),('global',0.),('local',1.)]:
        gate = ReliabilityAwareRoutingFusion(48,3,route_mode=mode)
        with torch.no_grad(): gate.log_temperatures.copy_(torch.tensor([.2,-.3]))
        r=gate(g,l,gf,lf)
        expected=(1-alpha)*torch.softmax(g/torch.exp(torch.tensor(.2)),1)+alpha*torch.softmax(l/torch.exp(torch.tensor(-.3)),1)
        torch.testing.assert_close(r['task_probs'],expected)
        assert bool((r['local_route_weight']==alpha).all())
    aux={'task_logits':b['task_logits'],'global_task_logits':g,'local_task_logits':l,
         'degradation_feature':gf,'local_degradation_feature':lf}
    labels=torch.tensor([1,2])
    default=blind_v3_routing_loss(aux,labels)
    old_default=old_loss.blind_v3_routing_loss(aux,labels)
    for k in default: torch.testing.assert_close(default[k],old_default[k],rtol=0,atol=0)
    fixed=blind_v3_routing_loss(aux,labels,teacher_mode='global')
    gradients=torch.autograd.grad(fixed['probability_consistency']+fixed['feature_consistency'],
                                  (g,l,gf,lf),allow_unused=True)
    assert gradients[0] is None or not bool(gradients[0].abs().sum())
    assert gradients[2] is None or not bool(gradients[2].abs().sum())
    assert gradients[1].abs().sum()>0 and gradients[3].abs().sum()>0
    assert fixed['global_teacher_fraction'].item()==1
    assert default['global_teacher_fraction'].item()==.5
    dataset=AllInOneDataset(plan['data_root'],manifest_path=plan['train_manifest'],
                            return_context=True,context_size=256,patch_size=256,noise_seed=42)
    samples=[dataset[dataset.task_indices[t][0]] for t in [0,1,2,0]]
    images,targets,tasks,contexts=[torch.stack([s[i] for s in samples]).cuda() for i in range(4)]
    reports=[]
    for run in plan['runs']:
        sys.argv=['train.py']+run['train_command'][4:]
        args=train.parse_args()
        assert not args.resume and not args.init_weights
        train.seed_everything(42)
        model,checkpoint,init=train.create_model(args,torch.device('cuda'))
        assert checkpoint is None and init is None
        train.configure_training_protocol(args,model)
        train.snapshot_data_protocol(args)
        model.train()
        optimizer=train.build_optimizer(model,args)
        scaler=torch.cuda.amp.GradScaler(init_scale=args.amp_init_scale)
        criterion=RestorationLoss(ssim_weight=args.ssim_weight,edge_weight=args.edge_weight,frequency_weight=args.frequency_weight)
        good=0
        records=[]
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                prediction,aux=model(images,task_prompt=tasks,routing_context=contexts,
                                     teacher_forcing=1. if step==0 else 0.,return_aux=True)
                restore,_=criterion(prediction,targets)
                route=blind_v3_routing_loss(aux,tasks,teacher_mode=args.v3_teacher_mode)
                loss=restore.mean()+.1*route['combined_cls']+args.v3_consistency_weight*route['probability_consistency']+args.feature_consistency_weight*route['feature_consistency']
            assert bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(loss))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            before=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            stepped=scaler.get_scale()>=before
            if stepped:
                assert bool(torch.isfinite(grad))
                good+=1
            if args.v3_route_mode!='learned':
                assert all(p.grad is None for p in model.hierarchical.routing_fusion.gate.parameters())
            records.append({'step':step+1,'loss':loss.item(),'updated':stepped,'scale':scaler.get_scale()})
            if good>=2: break
        assert good>=2, 'No successful training steps: '+run['name']
        path=out/(run['name']+'.pth')
        torch.save({'model_config':model.model_config,'model_state_dict':model.state_dict(),
                    'train_args':vars(args),'data_protocol':args.data_protocol},str(path))
        del model,optimizer
        loaded,ck,config=load_model_for_inference(str(path),torch.device('cuda'))
        assert config['v3_route_mode']==args.v3_route_mode
        result,routing=tiled_inference(loaded,images[:1,:,:96,:112],tile_size=64,overlap=16,amp_enabled=True,return_routing=True)
        assert result.shape==(1,3,96,112) and bool(torch.isfinite(result).all())
        assert bool(torch.isfinite(routing['task_probs']).all())
        # Resume must recover route and teacher even when CLI defaults differ.
        sys.argv=['train.py','--resume',str(path)]
        resumed_args=train.parse_args()
        resumed,ck,_=train.create_model(resumed_args,torch.device('cpu'))
        train.configure_training_protocol(resumed_args,resumed,ck)
        assert resumed_args.v3_route_mode==args.v3_route_mode
        assert resumed_args.v3_teacher_mode==args.v3_teacher_mode
        assert resumed_args.v3_consistency_weight==args.v3_consistency_weight
        report={'name':run['name'],'route':args.v3_route_mode,'teacher':args.v3_teacher_mode,
                'prob_weight':args.v3_consistency_weight,'feature_weight':args.feature_consistency_weight,
                'successful_steps':good,'steps':records,'checkpoint_reload':'passed','tile_inference':'passed','resume':'passed'}
        reports.append(report)
        print(json.dumps(report),flush=True)
        del loaded,resumed,ck
        torch.cuda.empty_cache()
    (out/'report.json').write_text(json.dumps({'status':'passed','default_equivalence':'exact',
         'shared_initialization':'exact','teacher_gradient':'passed','variants':reports},indent=2))


if __name__=='__main__': main()
