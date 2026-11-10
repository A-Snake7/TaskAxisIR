"""Frozen-checkpoint, controlled routing ablation. No training or test selection."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

OUT = Path('/root/ProMamba_Project/experiments/chapter6_supplement_20260917')
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT / 'code'))
import numpy as np
import torch
from scripts import evaluate_aio3_promptir as ev
from utils.checkpoint import load_model_for_inference
from utils.inference import tiled_inference as original_tiled
from utils.chapter6_inference import tiled_inference

CONDITIONS = ['derain', 'dehaze', 'denoise_sigma15', 'denoise_sigma25', 'denoise_sigma50']
TASKS = ['derain', 'dehaze', 'denoise']
MODES = ['shared_aggregate', 'independent_tiles']

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2))

def events(event, **kwargs):
    record = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=event, **kwargs)
    with (OUT / 'events.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)

def inputs():
    data = ROOT / 'data/test'
    for task in TASKS[:2]:
        index = ev.build_target_index(data / task / 'target')
        for path in ev.image_files(data / task / 'input'):
            target = ev.find_target(path, index, task)
            yield task, path.name, ev.load_rgb_uint8(path, 16), ev.load_rgb_uint8(target, 16)
    rng = np.random.RandomState(0)
    for sigma in [15, 25, 50]:
        for path in ev.image_files(data / 'denoise/target'):
            clean = ev.load_rgb_uint8(path, 16)
            noisy = np.clip(clean + rng.randn(*clean.shape) * sigma, 0, 255).astype(np.uint8)
            yield 'denoise_sigma' + str(sigma), path.name, noisy, clean

def routing_record(scope, condition, name, context, tiles=1):
    result = dict(scope=scope, condition=condition, image=name, task=2 if condition.startswith('denoise') else TASKS.index(condition), tile_count=tiles)
    for branch, key in [('global', 'global_task_probs'), ('local', 'local_task_probs'), ('fused', 'task_probs')]:
        probs = context[key].detach().float().cpu().reshape(-1).tolist()
        assert len(probs) == 3 and all(np.isfinite(probs)) and abs(sum(probs)-1) < 1e-5
        result[branch + '_probs'] = probs
        result[branch + '_prediction'] = int(np.argmax(probs))
    return result

def recall_summary(records):
    summaries = {}
    for scope in ['full_image', 'tiled_image_aggregate']:
        selected = [r for r in records if r['scope'] == scope]
        branches = {}
        for branch in ['global', 'local', 'fused']:
            confusion = np.zeros((3, 3), dtype=np.int64)
            for r in selected:
                confusion[r['task'], r[branch + '_prediction']] += 1
            recall = np.diag(confusion) / confusion.sum(axis=1)
            branches[branch] = dict(confusion_true_rows_pred_columns=confusion.tolist(),
                task_support=confusion.sum(axis=1).tolist(), per_task_recall=dict(zip(TASKS, recall.tolist())),
                macro_recall=float(recall.mean()), overall_accuracy=float(np.trace(confusion)/confusion.sum()),
                per_condition_recall={c: float(np.mean([r[branch + '_prediction']==r['task'] for r in selected if r['condition']==c])) for c in CONDITIONS})
        summaries[scope] = branches
    return summaries

def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    assert not (OUT / 'results.json').exists(), 'Results already exist'
    audit = json.loads((OUT / 'preparation_audit.json').read_text())
    assert ev.checkpoint_sha256(audit['weights']) == audit['checkpoint_sha256']
    for name, digest in audit['original_source_sha256'].items():
        assert ev.checkpoint_sha256(OUT / 'code' / name) == digest
    assert ev.checkpoint_sha256(OUT / 'code/utils/chapter6_inference.py') == audit['extension_sha256']
    # Verify the original evaluation dataset byte-for-byte before reuse.
    manifest_path = ROOT / '.testing_launches/taskaxisir_blind_v3_scratch_best84_aio3_fp32_full_20260911_s42/test_file_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for record in manifest:
        assert ev.checkpoint_sha256(record['path']) == record['sha256'], record['path']
    model, checkpoint, config = load_model_for_inference(audit['weights'], torch.device('cuda'))
    assert checkpoint['epoch'] == 84 and 'ema_state_dict' in checkpoint
    assert config['condition_mode'] == 'blind_v3' and not config['v3_fuse_prompt_condition']
    ev.self_test()
    events('preflight_passed', checkpoint_epoch=84, weights_kind='ema', dataset_files_verified=len(manifest))
    # Verify original shared path, single-tile identity, and isolated head-route changes.
    checks = []
    with torch.inference_mode():
        for shape in [(256,256), (192,208), (272,288)]:
            x = torch.rand(1,3,*shape, device='cuda')
            original = original_tiled(model,x,amp_enabled=False)
            shared, aux = tiled_inference(model,x,amp_enabled=False,return_routing=True)
            independent = tiled_inference(model,x,amp_enabled=False,independent_tile_routes=True)
            torch.testing.assert_close(shared, original, rtol=0, atol=0)
            if max(shape) <= 256:
                torch.testing.assert_close(shared, independent, rtol=1e-6, atol=1e-6)
            checks.append(dict(shape=shape, original_max_error=float((shared-original).abs().max()),
                               shared_independent_max_difference=float((shared-independent).abs().max())))
        # On a real multi-tile image, record forward inputs and confirm only head routes differ.
        condition, name, degraded, clean = next(inputs())
        x = ev.uint8_to_tensor(degraded, torch.device('cuda'))
        calls = []
        def hook(module, args, kwargs):
            calls.append((args[0].detach().clone(), kwargs['condition_override'].detach().clone(), kwargs['routing_probs_override'].detach().clone()))
        handle = model.register_forward_pre_hook(hook, with_kwargs=True)
        tiled_inference(model,x,amp_enabled=False)
        n = len(calls)
        tiled_inference(model,x,amp_enabled=False,independent_tile_routes=True)
        handle.remove()
        assert len(calls) == 2*n
        for a,b in zip(calls[:n],calls[n:]):
            assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
        checks.append(dict(real_image=name, tile_count=n, identical_tile_pixels_and_conditions=True))
    save('software_checks.json', checks)
    events('software_checks_passed')
    results = []
    routes = []
    counts = {}
    with (OUT / 'per_image_results.jsonl').open('x') as metrics_out, (OUT / 'routing_predictions.jsonl').open('x') as routes_out:
        with torch.inference_mode():
            for index, (condition, name, degraded, clean) in enumerate(inputs(), 1):
                assert degraded.shape == clean.shape
                x = ev.uint8_to_tensor(degraded, torch.device('cuda'))
                full_context = model.predict_task_context(x)
                full_record = routing_record('full_image', condition, name, full_context)
                routes.append(full_record)
                routes_out.write(json.dumps(full_record)+'\n')
                for mode in MODES:
                    restored, aux = tiled_inference(model,x,tile_size=256,overlap=32,amp_enabled=False,
                        return_routing=True,independent_tile_routes=(mode=='independent_tiles'))
                    psnr, ssim = ev.compute_metrics(ev.tensor_to_rgb_float(restored),clean.astype(np.float32)/255)
                    assert np.isfinite(psnr) and np.isfinite(ssim)
                    row = dict(mode=mode,condition=condition,image=name,psnr=psnr,ssim=ssim,tile_count=aux['tile_count'])
                    results.append(row)
                    metrics_out.write(json.dumps(row)+'\n')
                    if mode == 'shared_aggregate':
                        record = routing_record('tiled_image_aggregate',condition,name,aux,aux['tile_count'])
                        record['tile_predicted_classes'] = aux['tile_routes'].argmax(dim=1).cpu().tolist()
                        record['tile_disagreement'] = len(set(record['tile_predicted_classes'])) > 1
                        routes.append(record)
                        routes_out.write(json.dumps(record)+'\n')
                metrics_out.flush()
                routes_out.flush()
                counts[condition] = counts.get(condition,0)+1
                if index % 25 == 0:
                    events('progress',images_completed=index,total=804,condition=condition,image=name)
    assert list(counts.values()) == [100,500,68,68,68]
    summary = dict(checkpoint_sha256=audit['checkpoint_sha256'],weights=audit['weights'],weights_kind='ema',
        checkpoint_epoch=84, tile_size=256,overlap=32,precision='fp32',metric_domain='float',crop_base=16,noise_seed=0,
        sigma_order=[15,25,50],global_context='same full image for both modes',prompt_condition='global and identical',
        independent_definition='Per-tile fused head probability, retaining same full-image context; not the legacy local_blind_routing flag',
        shared_definition='Coverage-weighted per-tile fused probability shared by all output heads',conditions={},routing=recall_summary(routes))
    for mode in MODES:
        summary['conditions'][mode] = {}
        for condition in CONDITIONS:
            selected = [r for r in results if r['mode']==mode and r['condition']==condition]
            summary['conditions'][mode][condition] = dict(count=len(selected),psnr=float(np.mean([r['psnr'] for r in selected])),ssim=float(np.mean([r['ssim'] for r in selected])))
        summary['conditions'][mode]['avg5'] = {m:float(np.mean([summary['conditions'][mode][c][m] for c in CONDITIONS])) for m in ['psnr','ssim']}
    with (OUT / 'aggregation_per_image.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    save('results.json',summary)
    events('completed',images=804,modes=2)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        save('exit_status.json',dict(exit_code=1,traceback=traceback.format_exc()))
        raise
    else:
        save('exit_status.json',dict(exit_code=0,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
