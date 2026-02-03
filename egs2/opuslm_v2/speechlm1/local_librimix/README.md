# LibriMix Enhancement Evaluation Scripts

This directory contains evaluation scripts for LibriMix speech enhancement experiments.

## Overview

These scripts evaluate speech enhancement results from the OpusLM audio-to-audio model on the LibriMix dataset. The evaluation pipeline is organized into 2 stages:

1. **Stage 1: Prepare** - Parse inference results and create scp files
2. **Stage 2: Eval** - Compute metrics (stoi, estoi, si_snr, sdr, pesq, mos, wer)

## Quick Start

```bash
# Run full evaluation (both stages)
./local_librimix/run_eval.sh \
    --results_json exp/.../results.json \
    --dialogue_jsonl /path/to/dialogues_all.jsonl \
    --output_dir eval_results/ \
    --metrics "stoi estoi si_snr sdr pesq"

# With WER evaluation
./local_librimix/run_eval.sh \
    --results_json exp/.../results.json \
    --dialogue_jsonl /path/to/dialogues_all.jsonl \
    --output_dir eval_results/ \
    --metrics "stoi estoi si_snr sdr pesq wer" \
    --whisper_model base
```

## Scripts

### 1. `run_eval.sh` - Main Entry Point
Two-stage wrapper script for the evaluation pipeline.

**Usage:**
```bash
# Run both stages
./local_librimix/run_eval.sh \
    --results_json exp/.../results.json \
    --dialogue_jsonl /path/to/dialogues_all.jsonl \
    --output_dir eval_results/ \
    --metrics "stoi estoi si_snr sdr pesq"

# Run only stage 1 (prepare)
./local_librimix/run_eval.sh --stage 1 --stop_stage 1 \
    --results_json exp/.../results.json \
    --dialogue_jsonl /path/to/dialogues_all.jsonl \
    --output_dir eval_results/

# Run only stage 2 (eval) - assumes scp files exist
./local_librimix/run_eval.sh --stage 2 --stop_stage 2 \
    --output_dir eval_results/ \
    --metrics "stoi estoi si_snr sdr pesq wer"
```

**Options:**
- `--stage`: Start stage (default: 1)
- `--stop_stage`: Stop stage (default: 2)
- `--metrics`: Space-separated list of metrics (stoi, estoi, si_snr, sdr, pesq, mos, wer)
- `--whisper_model`: Whisper model size for WER (tiny, base, small, medium, large)
- `--device`: Device for ASR (cuda or cpu)

### 2. `prepare_eval_dir.py` - Stage 1
Prepare evaluation directory from inference results.

**Usage:**
```bash
python local_librimix/prepare_eval_dir.py \
    --results_json exp/.../results.json \
    --dialogue_jsonl /path/to/dialogues_all.jsonl \
    --output_dir eval_results/scp/
```

**Input:**
- `results.json`: Inference output `{example_id: [["assistant", "audio", "path"], ...]}`
- `dialogues_all.jsonl`: Original dialogue data with audio path metadata

**Output:**
- `enh.scp`: Enhanced audio paths (wav_id -> enhanced_path)
- `ref.scp`: Reference audio paths (wav_id -> reference_path)
- `mix.scp`: Mixed/noisy audio paths (wav_id -> mix_path)

### 3. `eval_enh.py` - Stage 2 (All-in-One Evaluation)
Compute speech enhancement metrics with configurable metric selection.

**Evaluation Mode:**
```bash
python local_librimix/eval_enh.py \
    --enh_scp eval_results/scp/enh.scp \
    --ref_scp eval_results/scp/ref.scp \
    --mix_scp eval_results/scp/mix.scp \
    --output_dir eval_results/ \
    --metrics stoi estoi si_snr sdr pesq

# With WER (transcript extraction from dialogue_jsonl)
python local_librimix/eval_enh.py \
    --enh_scp eval_results/scp/enh.scp \
    --ref_scp eval_results/scp/ref.scp \
    --output_dir eval_results/ \
    --metrics stoi wer \
    --dialogue_jsonl /path/to/dialogues_all.jsonl \
    --whisper_model base

# With MOS
python local_librimix/eval_enh.py \
    --enh_scp eval_results/scp/enh.scp \
    --ref_scp eval_results/scp/ref.scp \
    --output_dir eval_results/ \
    --metrics stoi mos
```

**Comparison Mode:**
```bash
# Compare multiple experiment results
python local_librimix/eval_enh.py --compare \
    --results_dir eval_step1000/ eval_step2000/ eval_step3000/ \
    --labels "1k" "2k" "3k" \
    --analyze --plot
```

**Available Metrics:**
| Metric | Description | Requirements |
|--------|-------------|--------------|
| `stoi` | Short-Time Objective Intelligibility | pystoi |
| `estoi` | Extended STOI | pystoi |
| `si_snr` | Scale-Invariant SNR | numpy |
| `sdr` | Signal-to-Distortion Ratio | mir_eval |
| `pesq` | Perceptual Evaluation of Speech Quality | pesq |
| `mos` | Pseudo-MOS (UTMOS, DNSMOS, PLCMOS) | versa |
| `wer` | Word Error Rate | openai-whisper, jiwer |

**Output:**
- `results.json`: Per-utterance metrics
- `summary.json`: Aggregate statistics (mean, std, min, max)
- `hypothesis.txt`: ASR transcriptions (if WER computed)
- `reference_text.txt`: Reference transcripts (if extracted)

### 4. `generate_report.py`
Generate evaluation reports in various formats.

**Usage:**
```bash
python local_librimix/generate_report.py \
    --results_dir eval_step1000/ eval_step2000/ \
    --labels "Step 1000" "Step 2000" \
    --output_dir reports/ \
    --format all  # markdown, latex, csv, or all
```

### 5. `batch_eval.sh`
Batch evaluation for multiple checkpoints in an experiment.

**Usage:**
```bash
./local_librimix/batch_eval.sh \
    --exp_dir exp/opuslm_v2_stage3_sft_librimix_enh-v3 \
    --test_set test100 \
    --metrics "stoi estoi si_snr sdr pesq wer"
```

## Requirements

Install required packages:
```bash
# Core metrics
pip install pystoi pesq mir_eval torchaudio tqdm numpy

# For WER computation
pip install openai-whisper jiwer

# For pseudo-MOS metrics (optional)
pip install versa

# For plots (optional)
pip install matplotlib
```

## Example Workflow

```bash
# Set paths
RESULTS_JSON=exp/opuslm_v2_stage3_sft_librimix_enh-v3/inference/inference_step_351881/dialogue_librimix_spk1_enh-v3_test100/results.json
DIALOGUE_JSONL=/mnt/home/xungong-andr-1766e0/prep/data/librimix_sft/data_mix_single/test100/spk1_enh/a2a_enh-v3/stage3_dialogues/dialogues_all.jsonl
OUTPUT_DIR=exp/opuslm_v2_stage3_sft_librimix_enh-v3/eval/test100

# Run full evaluation with all metrics including WER
./local_librimix/run_eval.sh \
    --results_json $RESULTS_JSON \
    --dialogue_jsonl $DIALOGUE_JSONL \
    --output_dir $OUTPUT_DIR \
    --metrics "stoi estoi si_snr sdr pesq wer"

# Or run stages separately:
# Stage 1: Prepare scp files
./local_librimix/run_eval.sh --stage 1 --stop_stage 1 \
    --results_json $RESULTS_JSON \
    --dialogue_jsonl $DIALOGUE_JSONL \
    --output_dir $OUTPUT_DIR

# Stage 2: Compute metrics
./local_librimix/run_eval.sh --stage 2 --stop_stage 2 \
    --output_dir $OUTPUT_DIR \
    --dialogue_jsonl $DIALOGUE_JSONL \
    --metrics "stoi estoi si_snr sdr pesq wer"

# Compare multiple checkpoints
python local_librimix/eval_enh.py --compare \
    --results_dir exp/.../eval_step1000 exp/.../eval_step2000 \
    --labels "Step 1k" "Step 2k" \
    --analyze
```
    --results_json $RESULTS_JSON \
    --dialogue_jsonl $DIALOGUE_JSONL \
    --output_dir $OUTPUT_DIR/scp

# Step 2: Compute enhancement metrics
python local_librimix/eval_enh.py \
    --enh_scp $OUTPUT_DIR/scp/enh.scp \
    --ref_scp $OUTPUT_DIR/scp/ref.scp \
    --mix_scp $OUTPUT_DIR/scp/mix.scp \
    --output_dir $OUTPUT_DIR

# Step 3: Analyze results
python local_librimix/analyze_results.py \
    --results_dir $OUTPUT_DIR \
    --plot
```

## Output Format

### results.json
```json
{
  "wav_id_1": {
    "stoi": 0.85,
    "estoi": 0.78,
    "si_snr": 12.5,
    "sdr": 10.2,
    "pesq": 3.2,
    "input_stoi": 0.65,
    "input_si_snr": 5.0,
    "stoi_improvement": 0.20,
    "si_snr_improvement": 7.5
  },
  ...
}
```

### summary.json
```json
{
  "stoi": {"mean": 0.85, "std": 0.05, "min": 0.70, "max": 0.95, "count": 100},
  "si_snr": {"mean": 12.5, "std": 2.0, "min": 8.0, "max": 18.0, "count": 100},
  ...
}
```
