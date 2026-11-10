# ProMamba

Prompt-conditioned restoration for deraining, dehazing, and denoising. The
original lightweight model remains available, while new runs default to the
three-scale hierarchical ProMamba.

## Environment

The server environment is:

- Conda environment: `airnet`
- Python 3.8
- PyTorch 2.0.1 + CUDA 11.8
- torchvision 0.15.2
- NVIDIA L40S

Run commands explicitly through the environment so non-interactive shells do
not depend on activation:

```bash
PY=/root/miniconda3/envs/airnet/bin/python
$PY train.py --help
```

## Data layout

```text
data/
  train/
    derain/{input,target}
    dehaze/{input,target}       # recursive subdirectories are supported
    denoise/target              # clean images; noise is generated online
  test/
    derain/{input,target}
    dehaze/{input,target}
    denoise/{input,target}
```

Filename alignment is a dry-run by default:

```bash
$PY align_all_data.py --base_dir data --mode test
# Review the output before:
$PY align_all_data.py --base_dir data --mode test --apply
```

The benchmark-aligned three-task training protocol is:

- Rain100L: 200 rainy/clean pairs.
- Historical RESIDE-OTS subset: 72,135 hazy images paired with 2,061 clean
  images. This was selected by numerical filename range and is retained only
  for archival reproducibility; it is not a paper-ready benchmark split.
- BSD400 + WED: 5,144 clean images with online Gaussian noise at sigma
  15/25/50.

The OTS mirror can contain SOTS samples in its `train` directory. Simply
ignoring the archive's `test` directory is therefore insufficient. New paper
runs must use a separately built scene-disjoint root and its audit report.

The former 13,990-image ITS training tree is preserved at
`data/optional/RESIDE_ITS` and is intentionally excluded from the comparable
All-in-One-3 protocol.

Downloads are kept outside `data/`; the training tree uses links so the
archives and extracted data are not duplicated:

```bash
# OTS is a resumable, checksummed split download.
nohup bash scripts/download_ots.sh > downloads/RESIDE_OTS/download.log 2>&1 &

$PY scripts/build_clean_aio3_dataset.py \
  --output-root data/aio3_available_ots_sots_disjoint_v1 \
  --hazy-source downloads/RESIDE_OTS/extracted/reside-outdoor/train/hazy \
  --hazy-target-source downloads/RESIDE_OTS/extracted/reside-outdoor/train/gt \
  --test-hazy-root data/test/dehaze/input \
  --test-target-root data/test/dehaze/target \
  --derain-source data/train/derain \
  --denoise-source data/train/denoise \
  --selection-label available_ots_sots_disjoint_v1

$PY scripts/audit_clean_aio3_dataset.py \
  --data-root data/aio3_available_ots_sots_disjoint_v1 \
  --test-root data/test \
  --verify-images

$PY scripts/create_split_manifests.py \
  --data_root data/aio3_available_ots_sots_disjoint_v1 \
  --output_dir splits/aio3_available_ots_sots_disjoint_v1_seed42

$PY scripts/audit_clean_aio3_split.py \
  --data-root data/aio3_available_ots_sots_disjoint_v1 \
  --test-root data/test \
  --dataset-audit data/aio3_available_ots_sots_disjoint_v1/audit_report.json \
  --train-manifest splits/aio3_available_ots_sots_disjoint_v1_seed42/train.csv \
  --val-manifest splits/aio3_available_ots_sots_disjoint_v1_seed42/val.csv \
  --output splits/aio3_available_ots_sots_disjoint_v1_seed42/split_audit.json

$PY scripts/verify_benchmark_data.py \
  --data-root data/test \
  --profile test \
  --strict
```

## Training

New runs use fixed-length balanced task sampling, AMP, three image scales,
prompt injection at each scale, four-direction row/column Mamba scans,
task-conditioned low/high-frequency fusion, and a
Charbonnier+SSIM+edge+frequency objective. Validation and inference use an
EMA model (`--ema_decay 0.999`). The default model has about 4.69M parameters.

```bash
$PY train.py \
  --data_root data/aio3_available_ots_sots_disjoint_v1 \
  --train_manifest splits/aio3_available_ots_sots_disjoint_v1_seed42_content_v2/train.csv \
  --val_manifest splits/aio3_available_ots_sots_disjoint_v1_seed42_content_v2/val.csv \
  --save_dir runs \
  --batch_size 4 \
  --patch_size 256 \
  --steps_per_epoch 2000 \
  --epochs 100 \
  --architecture hierarchical \
  --dim 48 \
  --stage_blocks 2 3 4
```

`--steps_per_epoch` is intentionally independent of OTS size. This prevents
the 200-image Rain100L task from being repeated hundreds of times just to
match the largest dataset.

The commands above have already created the current scene-disjoint manifests.
If you rebuild the clean root, recreate and audit the manifests before any
formal run. All degraded variants of one clean target remain in the same split,
and denoising validation expands each clean target to sigma 15, 25, and 50
without copying image files:

```bash
$PY scripts/create_split_manifests.py \
  --data_root data/aio3_available_ots_sots_disjoint_v1 \
  --output_dir splits/aio3_available_ots_sots_disjoint_v1_seed42 \
  --val_ratio 0.1 \
  --seed 42
```

For an architecture ablation, train the old-size network through
`--architecture lightweight --dim 64`.

### Blind routing baseline (no Task ID at inference)

`blind_soft` adds a lightweight degradation encoder. It predicts three task
probabilities from the shallow image features, mixes the existing 48-D task
embeddings with those probabilities, and adds a zero-initialized continuous
48-D learned degradation residual. This vector is not a supervised severity
estimate. The same probabilities
blend the three residual heads. Training labels come automatically from the
existing derain/dehaze/denoise dataset directories; inference needs only an
image.

Blind models can be trained directly from scratch; oracle pretraining is not
required. An optional warm start uses an independently verified clean oracle
checkpoint through `--init_weights`, not `--resume`, because the degradation
encoder has new parameters. Historical checkpoints trained from the contaminated
`data/train` must not initialize a clean benchmark run. The following is the
optional warm-start recipe for the `blind_soft` baseline:

```bash
$PY train.py \
  --data_root data/aio3_available_ots_sots_disjoint_v1 \
  --train_manifest splits/aio3_available_ots_sots_disjoint_v1_seed42/train.csv \
  --val_manifest splits/aio3_available_ots_sots_disjoint_v1_seed42/val.csv \
  --save_dir runs \
  --run_name taskaxisir_blind_soft \
  --architecture hierarchical \
  --condition_mode blind_soft \
  --init_weights runs/<clean_oracle_run>/best.pth \
  --router_warmup_epochs 3 \
  --backbone_lr 2e-5 \
  --condition_lr 5e-5 \
  --router_lr 2e-4 \
  --task_cls_weight 0.1 \
  --blind_teacher_forcing_start 1.0 \
  --blind_teacher_forcing_end 0.0 \
  --blind_teacher_forcing_epochs 20
```

The first router-warmup epochs freeze the pretrained restoration backbone. The
teacher-forcing schedule then linearly changes routing from the known training
label to predicted soft probabilities. Validation always uses fully blind
routing. `--resume` is still used only after a blind checkpoint already exists.
New training runs also batch forward/reverse axial scans when the bottleneck is
at most 64 pixels, which improved 256-pixel latency on the project L40. Old
checkpoints default to the original scan path for exact output compatibility;
use `--no_batch_reverse_scans` for the corresponding ablation.

### Blind-v2: global context and local gate correction

`--condition_mode blind_v2` requires `--architecture hierarchical`. Existing
`oracle` and `blind_soft` modes remain available and keep their original
restoration behavior. The v2 code is an implementation to be evaluated: no
completed formal v2 training or restoration-quality improvement is claimed.

The training dataset returns a local crop and a compact view of the full
degraded image. `prepare_routing_context` preserves aspect ratio, downsamples
only when necessary, and pads to `--routing_context_size` (default 256) using
reflection where valid or replication otherwise. Training and inference share
this preprocessing. For synthetic denoising, noise is sampled once on the full
image before cropping; the crop and context therefore use the same noisy
realization. Crop, target, and full degraded view share the sampled orientation
augmentation. The router never receives a clean target as its context.

A shared degradation encoder produces global and local features and task
probabilities. The global feature supplies the continuous condition; the global
probabilities mix task embeddings and output heads. The resulting global
condition controls every `PromptAffine`. Frequency, branch, and axial-direction
gates additionally receive a bounded local correction:

```text
c_gate = c_global + alpha * tanh(A(z_local))
alpha = --local_condition_scale  # default 0.1
```

`A` is LayerNorm followed by a linear layer whose weight and bias are initialized
to zero. Initially, local gate correction is exactly zero; each coordinate of
the correction is bounded by `alpha`. At tiled inference, global probabilities
and the continuous global condition are computed once from the original image
and reused across all tiles. Only the local correction is recomputed per tile.
These learned features are not explicit severity labels or an explicit
frequency/direction-energy estimator.

Training uses dataset task labels for auxiliary classification, but default
validation and inference do not use them to choose the restoration route. With
the default loss weights:

```text
L = L_restoration
    + 0.1 * (CE_global + 0.5 * CE_local)
    + 0.05 * KL(p_local || stop_gradient(p_global))
    + 0.01 * MSE(normalize(z_local), stop_gradient(normalize(z_global)))
```

The controls are `--task_cls_weight`, `--local_cls_weight`,
`--routing_consistency_weight`, and `--feature_consistency_weight`, respectively.
Consistency targets are detached on the global branch, and the routing losses
are evaluated in FP32 even under AMP. The encoder is shared, so local losses
still update its shared parameters. The existing teacher-forcing schedule
replaces label-based routing with predicted routing over the first 20 epochs
by default; it never applies to validation or normal inference.

### Blind-v3: reliability-aware global/detail routing

`--condition_mode blind_v3` keeps the v2 restoration backbone and, by default,
the global-only `PromptAffine` condition, but fixes an important routing limitation: the local
classifier now participates in task-head selection instead of being used only
for gate diagnostics. Calibrated global and local probabilities are combined
by a small reliability gate. The gate observes probabilities, confidence,
margin, normalized entropy, Jensen-Shannon disagreement, and normalized
degradation features. Its initial local weight is controlled by
`--v3_local_route_prior` (default 0.5).

```text
p_g = softmax(z_global / T_global)
p_l = softmax(z_local  / T_local)
a = sigmoid(MLP(reliability statistics))
p_fused = (1 - a) * p_g + a * p_l
```

The complete-image global condition is still computed once. Tiled inference
performs a cheap routing-only pass in which every tile is fused with that
global view exactly like a training crop. The per-tile fused probabilities are
then coverage-weighted into one image-level task-head route, which is reused
with one cached condition for all restoration tiles. It therefore avoids both
tile-dependent task-head seams and a separate untrained feature-aggregation
distribution.

The default (`--no_v3_fuse_prompt_condition`) is the conservative head-routing
ablation: the fused route selects the output heads while `PromptAffine` retains
the full-image condition inherited from v2. `--v3_fuse_prompt_condition` also
uses that one image-level fused route for the discrete task-token part of every
`PromptAffine`; its continuous degradation feature still comes from the global
view. Both modes cache the condition once, so no tile receives a different
task token.

V3 removes the fixed `global -> local` teacher assumption. Global, local, and
fused routes each receive task-label cross-entropy. For consistency, the view
assigning the larger probability to the known training task becomes the
detached teacher on each sample; the other view receives the consistency
gradient. Validation records per-task global/local/fused confusion matrices,
macro routing accuracy, learned local route weight, and output-versus-input
PSNR gain statistics including negative-gain counts.

For v3, `--best_selection_metric auto` saves `best.pth` by
`val_avg5_proxy_psnr = (PSNR_rain + PSNR_haze + 3 * PSNR_noise) / 5`, matching
the weight pattern of the formal one-rain, one-haze, three-noise Avg-5 report.
`val_macro_psnr` is still recorded for a task-balanced paper comparison.
Use `--best_selection_metric macro_psnr` only as an explicit ablation.

Warm-start v3 from a verified v2 checkpoint with `--init_weights`, never with
`--resume`. The loader verifies that only the new `routing_fusion` tensors are
missing; optimizer, scheduler, and AMP scaler states start fresh:

```bash
$PY train.py \
  --data_root data/aio3_available_ots_sots_disjoint_v1 \
  --train_manifest splits/aio3_available_ots_sots_disjoint_v1_seed42_content_v2/train.csv \
  --val_manifest splits/aio3_available_ots_sots_disjoint_v1_seed42_content_v2/val.csv \
  --save_dir runs \
  --run_name taskaxisir_blind_v3_fusion_screen \
  --architecture hierarchical \
  --condition_mode blind_v3 \
  --init_weights runs/taskaxisir_blind_v2_clean_content_20260828_s42/best.pth \
  --router_warmup_epochs 2 \
  --v3_consistency_weight 0.01 \
  --feature_consistency_weight 0.01 \
  --val_inference tile \
  --val_tile_size 256 \
  --val_overlap 32 \
  --best_selection_metric avg5_proxy_psnr \
  --amp_init_scale 1024 \
  --batch_size 4 \
  --patch_size 256 \
  --steps_per_epoch 500 \
  --epochs 20
```

This is a short routing screen, not the final budget-matched paper comparison.
Keep v2 unchanged as the ablation baseline. Do not add the identity/safety gate
until oracle-head and oracle-condition diagnostics show how many harmful
light-rain outputs remain after routing is corrected.

#### Clean training from scratch

Run from `/root/ProMamba_Project` with the `PY` environment variable defined
above. Use a new run name and do not pass historical weights:

```bash
$PY train.py \
  --data_root data/aio3_available_ots_sots_disjoint_v1 \
  --train_manifest splits/aio3_available_ots_sots_disjoint_v1_seed42/train.csv \
  --val_manifest splits/aio3_available_ots_sots_disjoint_v1_seed42/val.csv \
  --save_dir runs \
  --run_name taskaxisir_blind_v2_clean \
  --architecture hierarchical \
  --condition_mode blind_v2 \
  --router_warmup_epochs 0 \
  --backbone_lr 2e-4 \
  --condition_lr 2e-4 \
  --router_lr 2e-4 \
  --routing_context_size 256 \
  --local_condition_scale 0.1 \
  --val_inference tile \
  --val_tile_size 256 \
  --val_overlap 32 \
  --batch_size 4 \
  --patch_size 256 \
  --steps_per_epoch 2000 \
  --epochs 100
```

Scratch training must set `--router_warmup_epochs 0`: otherwise router-only
warmup would freeze a random restoration backbone. Specify all three learning
rates as above; `--lr` alone does not set the blind parameter-group rates. The
command describes a training configuration, not a completed run or a guaranteed
convergence schedule.

V2 requires both training and validation manifests. It records absolute data
paths and manifest SHA256 hashes and restores and verifies them on resume, so
omitting data arguments cannot silently select the old `data/train`. Keep the
dataset and split audit reports with each experiment; manifest hashes alone do
not prove that image contents are unchanged or that a split is leakage-free.
The current clean root is the scene-disjoint subset available on this server,
not an exact reproduction of the official 72,135-image OTS training list.

An optional clean-oracle warm start uses the same v2 command with
`--init_weights runs/<verified_clean_oracle_run>/best.pth`,
`--router_warmup_epochs 3`, `--backbone_lr 2e-5`, and `--condition_lr 5e-5`;
keep `--router_lr 2e-4`. Do not use a checkpoint trained on the old contaminated
data. Comparisons with scratch baselines must disclose and count oracle
pretraining steps as part of the warm-start model's total training budget.

#### Validation and ablations

For v2, `--val_inference auto` resolves to full-image tiled validation with batch
size 1, tile size 256, and overlap 32. The best checkpoint is selected by EMA
validation macro-task PSNR, using predicted routing only. Logs also record
global task accuracy, entropy, 15-bin ECE, a task confusion matrix, and
global/local Jensen-Shannon divergence. In tiled validation,
`val_local_task_accuracy` classifies each image using the mean local probability
vector across its tiles; it is not a per-tile accuracy statistic.

| Experiment | Controls | Interpretation |
| --- | --- | --- |
| Existing blind baseline | `--condition_mode blind_soft` | Preserve the baseline; explicitly match validation protocol and training budget. |
| No local gate correction | `--local_condition_scale 0` | Retain global conditioning and auxiliary view losses. |
| No view consistency | `--routing_consistency_weight 0 --feature_consistency_weight 0` | Retain both classification losses and local correction. |
| No local classification | `--local_cls_weight 0` | Global CE and consistency losses remain active. |
| No frequency branch | `--no_frequency_branch` | Remove the frequency branch; other conditioning remains. |
| Uniform direction fusion | `--direction_fusion mean` | Replace learned scan-direction weights with their mean. |
| No feature conditioning | `--no_task_condition` | Disable feature modulation and gate conditions, but still mix task heads. |
| Unconditional restoration path | `--no_task_condition --shared_head` | Also remove task-dependent head mixing; auxiliary router losses remain unless their weights are set to zero. |

Use `--val_inference crop` only as an explicitly named validation-protocol
ablation. `--val_inference full` evaluates full-image restoration without
tiling and can require substantially more memory. For fair model comparisons,
hold the audited split, number of optimizer steps, crop size, precision,
validation and final-test inference protocols fixed. A larger module count is
not evidence that an innovation improves restoration.

Do not use the final test set to choose the best checkpoint. Use the generated
validation manifest; its targets are disjoint from the training manifest.

Resume defaults to a new timestamped run directory and restores the architecture
stored in the checkpoint. For v2 it also restores the bound data, validation,
and loss settings and checks manifest hashes before continuing:

```bash
$PY train.py \
  --resume runs/taskaxisir_blind_v2_clean/latest.pth \
  --epochs 200 \
  --save_dir runs
```

`--epochs` is the new total epoch target, not an additional epoch count. Legacy
`oracle`/`blind_soft` resumes still require explicitly supplying their original
data and manifest arguments. Use a new initialization run, not resume, when
intentionally changing the training data or architecture.

The June 2026 checkpoint remains compatible:

```text
checkpoints/promamba_legacy_epoch_0130.pth
```

Older checkpoint families were trained with different model definitions and
must be used with their archived source.

## Evaluation

Full-image evaluation uses overlapping tiles and writes PNG outputs plus JSON
and CSV metrics:

```bash
$PY test.py \
  --weights checkpoints/promamba_legacy_epoch_0130.pth \
  --input_dir data/test/derain/input \
  --target_dir data/test/derain/target \
  --task derain \
  --save_dir evaluations/derain
```

Run all tasks:

```bash
bash run_all_tests.sh checkpoints/promamba_legacy_epoch_0130.pth
```

Single image:

```bash
$PY test_single.py \
  --weights runs/taskaxisir_blind_v2_clean/best.pth \
  --img_path example.png \
  --save_path restored.png \
  --tile_size 256 \
  --overlap 32 \
  --no-amp
```

For a blind checkpoint, `--task` in `test.py` identifies the benchmark and is
not passed into the model. `test_single.py` needs no task argument. Use
`--oracle_routing --task <id>` only for the oracle-routing ablation. Legacy
oracle checkpoints still require Task IDs: 0 for derain, 1 for dehaze, and 2
for denoise.

Tiled blind inference predicts one soft routing distribution from a resized
view of the complete image and reuses it for all tiles. This avoids inconsistent
task decisions at tile boundaries and adds little memory. Pass
`--local_blind_routing` only for the per-tile routing ablation.

For v2, this ablation recomputes the nominal global context from each tile as
well as its local correction; it deliberately removes image-level context
sharing. Compare tile-size sensitivity with `--tile_size 128`, `256`, or `512`
and explicitly fix precision and overlap. Oracle-routing ablation for a single
dehazing image adds `--oracle_routing --task 1` to the single-image command;
`test.py` uses task names instead, e.g. `--oracle_routing --task dehaze`.
The v2 routing diagnostics keep predicted probabilities separate from the
label-forced probabilities used by the heads.

On the project L40, precision defaults to FP32 for 256-pixel tiles and AMP for
tiles of 512 pixels or larger. Explicit `--amp` and `--no-amp` override this
measured automatic choice. Blind evaluation CSV files also contain the three
task probabilities, predicted task, and routing confidence; JSON summaries add
task accuracy and a confusion matrix whenever `--task` provides ground truth.

## Run outputs

Each training run contains:

```text
runs/<timestamp>/
  config.json
  training.log
  metrics.csv
  latest.pth
  best.pth          # only when validation is configured
  epoch_XXXX.pth
```

Checkpoints are saved atomically and contain model configuration, optimizer,
scheduler, scaler, arguments, and RNG state.

## Blind-v2 software verification

Run these checks in the project's Python environment before a formal run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -B -m unittest discover -s tests -p 'test_*.py' -v
$PY -B scripts/blind_v2_smoke_test.py --output diagnostics/blind_v2_synthetic_smoke.json
```

The integration smoke check uses three AMP optimizer steps with the full-size
Mamba model, then a tiny synthetic model to exercise training, resume without
repeating data paths, blind single/batch inference, and manifest-change
rejection. Toy data and toy checkpoints live in an automatically removed
temporary directory. Its PSNR/loss values are software diagnostics only, not
paper results. `--skip_gpu` skips the full-size GPU check; it does not force the
remaining toy training onto CPU if CUDA is available.
