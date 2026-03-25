# SpeechLM Agent Guide (Model Build, Setup, Inference)

## Scope
This guide is for:
- code under `/Users/insight/Downloads/espnet/espnet2/speechlm`
- recipe under `/Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1`
- target configs:
  - train: `/Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1/conf/train_stage3_qwen3_base.yaml`
  - infer: `/Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1/conf/inference_pt.yaml`

## Primary Entrypoints
- Train CLI: `/Users/insight/Downloads/espnet/espnet2/speechlm/bin/train.py`
- Inference CLI: `/Users/insight/Downloads/espnet/espnet2/speechlm/bin/inference.py`
- Stats builder: `/Users/insight/Downloads/espnet/espnet2/speechlm/bin/prepare_length_stats.py`
- Dataset manifest builder: `/Users/insight/Downloads/espnet/espnet2/speechlm/bin/prepare_dataset_json.py`
- Stage scripts:
  - `/Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1/launch_opuslm_stage1_warmup.sh`
  - `/Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1/launch_opuslm_stage2_pretrain.sh`
  - `/Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1/launch_opuslm_stage3_sft.sh`

## Model Build Flow
`train.py` builds the model from YAML config in this chain:
1. `job_type` -> `/Users/insight/Downloads/espnet/espnet2/speechlm/model/__init__.py` (`speechlm` -> `SpeechLMJobTemplate`)
2. `SpeechLMJobTemplate` in `/Users/insight/Downloads/espnet/espnet2/speechlm/model/speechlm/speechlm_job.py`:
   - builds `multimodal_io` (`text`, `discrete_audio`, `continuous_audio`)
   - builds vocab and stream intervals
   - builds preprocessor and model
3. `model.model_choice: parallel` -> `ParallelHFModel` in
   `/Users/insight/Downloads/espnet/espnet2/speechlm/model/speechlm/lm/parallel.py`:
   - loads HF backbone from `model_hf_tag`
   - rebuilds embeddings/lm_head for multimodal vocab
   - supports multimodal decoding and modality-constrained inference

## Full Setup (Environment + Runtime)
Run from ESPnet root:

```bash
cd /Users/insight/Downloads/espnet
```

Suggested env setup:

```bash
cd tools
bash setup_anaconda.sh miniconda3 dev 3.11
source activate_python.sh
```

Install SpeechLM deps:

```bash
cd /Users/insight/Downloads/espnet/espnet2/speechlm
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirement.txt
pip install flash-attn --no-build-isolation
```

Recipe runtime setup (important):

```bash
cd /Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1
. ./path.sh
. ./cmd.sh
```

Notes:
- `path.sh` exports `PYTHONPATH` and `ESPNET_DATASET_REGISTRY`.
- Registered specifiers like `dialogue:librispeech_test_clean` are resolved from `ESPNET_DATASET_REGISTRY`.
- If dataset registry is not available in your environment, set it manually to your YAML registry files.

## Data/Manifest Setup Requirements
1. Build per-dataset manifests (`data_entry` + `samples`) with `prepare_dataset_json.py`.
2. Build length stats with `prepare_length_stats.py`.
3. Ensure stats files exist for each used `(task, data_name)`:
   - `exp/stats_qwen3/stats_<task>_<data_name>.jsonl`

If stats are missing, training fails in `DataIteratorFactory`.

## Stage-3 Training (Using Provided Train Config)
Config:
- `/Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1/conf/train_stage3_qwen3_base.yaml`

Important fields:
- `job_type: speechlm`
- backbone/tag: `Qwen/Qwen3-8B-Base`
- trainer deepspeed config: `conf/deepspeed_stage3.json`
- `data_loading.batchfy_method: pack`

Typical training command pattern:

```bash
cd /Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=localhost \
  --master_port=8888 \
  ../../../espnet2/speechlm/bin/train.py \
    --train-registered-specifier "<space-separated task:name[:factor]>" \
    --valid-registered-specifier "<space-separated task:name[:factor]>" \
    --train-config conf/train_stage3_qwen3_base.yaml \
    --stats-dir exp/stats_qwen3 \
    --output-dir exp/opuslm_v2_stage3_sft_qwen3_base \
    --resume-path exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_260000 \
    --save-loader-state \
    --wandb-mode offline
```

Resume behavior in `DeepSpeedTrainer`:
- prefer explicit `--resume-path`
- else auto-picks latest `output_dir/checkpoints/step_*`

## Inference (Using Provided `inference_pt.yaml`)
Config:
- `/Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1/conf/inference_pt.yaml`

This config enforces text-only output (`enforce_modality: ["text"]`).

Single-job example:

```bash
cd /Users/insight/Downloads/espnet/egs2/opuslm_v2/speechlm1

python ../../../espnet2/speechlm/bin/inference.py \
  --rank 1 \
  --world-size 1 \
  --train-config exp/opuslm_v2_stage3_sft_qwen3_base/train.yaml \
  --inference-config conf/inference_pt.yaml \
  --model-checkpoint exp/opuslm_v2_stage3_sft_qwen3_base/checkpoints/step_272500/global_step272500/mp_rank_00_model_states.pt \
  --output-dir exp/opuslm_v2_stage3_sft_qwen3_base/inference/inference_pt_step_272500 \
  --test-registered-specifier "dialogue:librispeech_test_clean" \
  --num-workers 3
```

Inference behavior details:
- Exactly one of `--test-registered-specifier` or `--test-unregistered-specifier` is required.
- `--rank` is treated as 1-based by script logic.
- Output goes to `<output-dir>/<specifier_with_colons_replaced>/inference_rank*/results.json`.
- Audio segments (if modality includes audio) are written as `.wav`.

## Known Recipe Issues To Fix
1. Launch scripts currently pass `--num-worker` to inference.
   - Correct flag is `--num-workers`.
   - Affects:
     - `launch_opuslm_stage1_warmup.sh`
     - `launch_opuslm_stage2_pretrain.sh`
     - `launch_opuslm_stage3_sft.sh`
2. `launch_opuslm_stage3_sft.sh` defaults to `conf/inference_sft.yaml`.
   - For your requested text-only inference, change to `conf/inference_pt.yaml`.

## Quick Debug Checklist
- `ModuleNotFoundError/espnet2`:
  - ensure `. ./path.sh` is sourced.
- `Dataset not found in registry`:
  - verify `ESPNET_DATASET_REGISTRY` points to valid YAML registries.
- `Statistics file not found`:
  - run `prepare_length_stats.py` for all used train/valid specifiers.
- `unrecognized arguments: --num-worker`:
  - use `--num-workers`.
