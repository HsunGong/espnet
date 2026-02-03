#!/usr/bin/env python3
"""
Evaluate speech enhancement results with configurable metrics.

Compute various metrics (configurable via --metrics):
- stoi: Short-Time Objective Intelligibility
- estoi: Extended STOI
- si_snr: Scale-Invariant Signal-to-Noise Ratio
- sdr: Signal-to-Distortion Ratio
- pesq: Perceptual Evaluation of Speech Quality
- mos: Pseudo-MOS using UTMOS, DNSMOS, PLCMOS
- wer: Word Error Rate using ASR models

This script also supports:
- Extracting transcripts from dialogue_jsonl for WER computation
- Comparison/analysis of multiple result directories

Usage:
    # Basic evaluation with all signal metrics
    python eval_enh.py --enh_scp enh.scp --ref_scp ref.scp --output_dir results/ \\
        --metrics stoi estoi si_snr sdr pesq

    # With WER (requires reference text or dialogue_jsonl)
    python eval_enh.py --enh_scp enh.scp --ref_scp ref.scp --output_dir results/ \\
        --metrics stoi wer --dialogue_jsonl dialogues.jsonl

    # Compare multiple result directories
    python eval_enh.py --compare --results_dir exp1/eval exp2/eval
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Available metrics
SIGNAL_METRICS = ["stoi", "estoi", "si_snr", "sdr", "pesq"]
MOS_METRICS = ["mos", "utmos", "dnsmos", "plcmos"]  # expands to utmos, dnsmos, plcmos
WER_METRICS = ["wer"]  # expands to wer, cer
VERSA_METRICS = ["speaker", "squim", "discrete"]
ALL_METRICS = SIGNAL_METRICS + MOS_METRICS + WER_METRICS + VERSA_METRICS


# =============================================================================
# Utility Functions
# =============================================================================


# =============================================================================
# Utility Functions
# =============================================================================

def load_scp(scp_path: str) -> Dict[str, str]:
    """Load a wav.scp file into a dictionary."""
    mapping = {}
    with open(scp_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def load_text(text_path: str) -> Dict[str, str]:
    """Load a text file into a dictionary (format: wav_id text)."""
    mapping = {}
    with open(text_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
            elif len(parts) == 1:
                mapping[parts[0]] = ""
    return mapping


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample if necessary.
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        return waveform.numpy().squeeze(), sr
    except ImportError:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio, sr


# =============================================================================
# Transcript Extraction (from dialogue_jsonl)
# =============================================================================

def normalize_text_for_wer(text: str) -> str:
    """Normalize text for WER computation."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def extract_transcripts_from_dialogue(dialogue_path: str) -> Dict[str, str]:
    """
    Extract transcripts from dialogues_all.jsonl.
    
    The metadata contains 'source_texts' field with transcriptions.
    
    Args:
        dialogue_path: Path to dialogues_all.jsonl
        
    Returns:
        Dict mapping wav_id to transcript
    """
    transcripts = {}
    
    with open(dialogue_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                continue
            
            metadata = data.get("metadata", {})
            wav_id = metadata.get("idx", "")
            source_texts = metadata.get("source_texts", [])
            
            if not wav_id:
                continue
            
            # source_texts[0] is typically the main speech transcription
            if source_texts and source_texts[0]:
                text = source_texts[0].upper()  # LibriSpeech convention
                text = ' '.join(text.split())
                transcripts[wav_id] = text
    
    return transcripts


# =============================================================================
# Versa Metrics
# =============================================================================

def compute_stoi(ref: np.ndarray, enh: np.ndarray, sr: int = 16000, extended: bool = False) -> float:
    """Compute STOI or ESTOI score using Versa."""
    try:
        from versa import stoi_metric, estoi_metric
        if extended:
            res = estoi_metric(enh, ref, sr)
            return float(res["estoi"])
        else:
            res = stoi_metric(enh, ref, sr)
            return float(res["stoi"])
    except Exception as e:
        logger.warning(f"Error computing STOI (extended={extended}): {e}")
        return float('nan')


def compute_si_snr(ref: np.ndarray, enh: np.ndarray) -> float:
    """Compute Scale-Invariant SNR using Versa."""
    try:
        from versa import signal_metric
        res = signal_metric(enh, ref)
        return float(res["si_snr"])
    except Exception as e:
        logger.warning(f"Error computing SI-SNR: {e}")
        return float('nan')


def compute_pesq(ref: np.ndarray, enh: np.ndarray, sr: int = 16000, mode: str = 'wb') -> float:
    """Compute PESQ score using Versa."""
    try:
        from versa import pesq_metric
        # versa's pesq_metric assumes 'wb' (wideband) if fs=16000 usually
        res = pesq_metric(enh, ref, sr)
        return float(res["pesq"])
    except Exception as e:
        logger.warning(f"Error computing PESQ: {e}")
        return float('nan')


def compute_sdr(ref: np.ndarray, enh: np.ndarray) -> float:
    """Compute Signal-to-Distortion Ratio using Versa."""
    try:
        from versa import signal_metric
        res = signal_metric(enh, ref)
        return float(res["sdr"])
    except Exception as e:
        logger.warning(f"Error computing SDR: {e}")
        return float('nan')

def compute_pseudo_mos(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Compute pseudo-MOS using various models.
    
    Returns dict with keys: utmos, dnsmos, plcmos
    """
    results = {}
    try:
        from versa import (
            pseudo_mos_metric,
            pseudo_mos_setup,
        )
        
        predictor_dict, predictor_fs = pseudo_mos_setup(
            use_gpu=True,
            predictor_types=["utmos", "dnsmos", "plcmos"],
            predictor_args={
                "utmos": {"fs": 16000},
                "dnsmos": {"fs": 16000},
                "plcmos": {"fs": 16000},
            },
        )
        
        mos_scores = pseudo_mos_metric(
            audio.astype(np.float32),
            sr,
            predictor_dict=predictor_dict,
            predictor_fs=predictor_fs,
            use_gpu=True,
        )
        results.update(mos_scores)
            
    except ImportError:
        logger.warning("Versa not installed, skipping pseudo-MOS computation")
    except Exception as e:
        logger.warning(f"Error computing pseudo-MOS: {e}")
    
    return results

def compute_wer_versa(wer_utils, ref: str, enh: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """Compute WER and CER using Versa (Whisper)."""
    results = {}
    try:
        from versa import whisper_levenshtein_metric
        # versa metric returns detailed counts
        res = whisper_levenshtein_metric(wer_utils, enh, ref, sr)
        
        # Calculate WER
        wer_del = res.get("whisper_wer_delete", 0)
        wer_ins = res.get("whisper_wer_insert", 0)
        wer_rep = res.get("whisper_wer_replace", 0)
        wer_eq  = res.get("whisper_wer_equal", 0)
        ref_len_w = wer_del + wer_rep + wer_eq
        
        wer = (wer_del + wer_ins + wer_rep) / ref_len_w if ref_len_w > 0 else float('nan')
        
        # Calculate CER
        cer_del = res.get("whisper_cer_delete", 0)
        cer_ins = res.get("whisper_cer_insert", 0)
        cer_rep = res.get("whisper_cer_replace", 0)
        cer_eq  = res.get("whisper_cer_equal", 0)
        ref_len_c = cer_del + cer_rep + cer_eq
        
        cer = (cer_del + cer_ins + cer_rep) / ref_len_c if ref_len_c > 0 else float('nan')
        
        results["wer"] = wer
        results["cer"] = cer
        results["hypothesis"] = res.get("whisper_hyp_text", "")
        
    except Exception as e:
        logger.warning(f"Error computing WER (Versa): {e}")
        results["wer"] = float('nan')
        results["cer"] = float('nan')
        results["hypothesis"] = ""
        
    return results


def compute_speaker_similarity(model, ref: np.ndarray, enh: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """Compute Speaker Similarity using Versa."""
    try:
        from versa import speaker_metric
        # versa expects (model, pred_x, gt_x, fs)
        res = speaker_metric(model, enh, ref, sr)
        # unwrap single value if needed, but dict comprehension handles it
        return {k: float(v) for k, v in res.items()}
    except Exception as e:
        logger.warning(f"Error computing Speaker Similarity: {e}")
        return {"spk_similarity": float('nan')}


def compute_squim(ref: np.ndarray, enh: np.ndarray, sr: int = 16000, compute_ref: bool = True, compute_no_ref: bool = True) -> Dict[str, float]:
    """Compute Torchaudio-Squim metrics."""
    results = {}
    from versa import squim_metric, squim_metric_no_ref
    
    # Reference-based
    if compute_ref and ref is not None:
        try:
             res = squim_metric(enh, ref, sr)
             results.update({k: float(v) for k, v in res.items()})
        except Exception as e:
            logger.warning(f"Error computing Squim Ref: {e}")
    
    # Reference-less
    if compute_no_ref:
        try:
            res = squim_metric_no_ref(enh, sr)
            results.update({k: float(v) for k, v in res.items()})
        except Exception as e:
            logger.warning(f"Error computing Squim No-Ref: {e}")
            
    return results


def compute_discrete(predictors, ref: np.ndarray, enh: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """Compute Discrete Speech metrics."""
    try:
        from versa import discrete_speech_metric
        res = discrete_speech_metric(predictors, enh, ref, sr)
        return {k: float(v) for k, v in res.items()}
    except Exception as e:
        logger.warning(f"Error computing Discrete metrics: {e}")
        return {}


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_pair(
    enh_path: str,
    ref_path: str,
    mix_path: Optional[str] = None,
    reference_text: Optional[str] = None,
    target_sr: int = 16000,
    metrics: List[str] = None,
    asr_model: Optional[object] = None,
    speaker_model: Optional[object] = None,
    discrete_predictors: Optional[object] = None,
) -> Dict[str, float]:
    """
    Evaluate a single enhanced/reference audio pair.
    
    Args:
        enh_path: Path to enhanced audio
        ref_path: Path to reference audio
        mix_path: Path to mixed/noisy audio (optional)
        reference_text: Reference transcript for WER (optional)
        target_sr: Target sample rate
        metrics: List of metrics to compute
        asr_model: ASR model instance for WER computation
        speaker_model: Speaker model for similarity metric
        discrete_predictors: Discrete speech predictors
    
    Returns:
        Dictionary of metric names to values
    """
    if metrics is None:
        metrics = SIGNAL_METRICS
    
    results = {}
    
    # Load audio
    enh_audio, _ = load_audio(enh_path, target_sr)
    ref_audio, _ = load_audio(ref_path, target_sr)
    
    # Signal quality metrics
    if "stoi" in metrics:
        results["stoi"] = compute_stoi(ref_audio, enh_audio, target_sr, extended=False)
    
    if "estoi" in metrics:
        results["estoi"] = compute_stoi(ref_audio, enh_audio, target_sr, extended=True)
    
    if "si_snr" in metrics:
        results["si_snr"] = compute_si_snr(ref_audio, enh_audio)
    
    if "sdr" in metrics:
        results["sdr"] = compute_sdr(ref_audio, enh_audio)
    
    if "pesq" in metrics:
        pesq_score = compute_pesq(ref_audio, enh_audio, target_sr)
        if not np.isnan(pesq_score):
            results["pesq"] = pesq_score
    
    # Input metrics for improvement calculation
    if mix_path and os.path.exists(mix_path):
        mix_audio, _ = load_audio(mix_path, target_sr)
        if "stoi" in metrics:
            results["input_stoi"] = compute_stoi(ref_audio, mix_audio, target_sr, extended=False)
            if not np.isnan(results.get("stoi", float('nan'))) and not np.isnan(results["input_stoi"]):
                results["stoi_improvement"] = results["stoi"] - results["input_stoi"]
        if "si_snr" in metrics:
            results["input_si_snr"] = compute_si_snr(ref_audio, mix_audio)
            if not np.isnan(results.get("si_snr", float('nan'))) and not np.isnan(results["input_si_snr"]):
                results["si_snr_improvement"] = results["si_snr"] - results["input_si_snr"]
    
    # Pseudo-MOS metrics
    if "mos" in metrics:
        results.update(compute_pseudo_mos(enh_audio, target_sr))
    
    # WER metrics
    if "wer" in metrics and asr_model is not None:
        try:
            # asr_model is wer_utils from versa
            # If reference text is missing, use empty string to at least capture hypothesis
            ref = reference_text if reference_text else ""
            wer_res = compute_wer_versa(asr_model, ref, enh_audio, target_sr)
            results.update(wer_res)
        except Exception as e:
            logger.warning(f"Error in WER computation: {e}")

    # Versa Metrics
    if "speaker" in metrics and speaker_model is not None:
        results.update(compute_speaker_similarity(speaker_model, ref_audio, enh_audio, target_sr))

    if "squim" in metrics:
        results.update(compute_squim(ref_audio, enh_audio, target_sr))
        
    if "discrete" in metrics and discrete_predictors is not None:
        results.update(compute_discrete(discrete_predictors, ref_audio, enh_audio, target_sr))
    
    return results


# =============================================================================
# Analysis Functions
# =============================================================================

def load_results(results_dir: str) -> Dict:
    """Load results from a results.json file."""
    results_path = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with open(results_path, 'r') as f:
        return json.load(f)


def load_summary(results_dir: str) -> Dict:
    """Load summary from a summary.json file."""
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with open(summary_path, 'r') as f:
        return json.load(f)


def compare_checkpoints(
    results_dirs: List[str],
    labels: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> None:
    """Compare evaluation results across multiple checkpoints/experiments."""
    if labels is None:
        labels = [os.path.basename(d) for d in results_dirs]
    
    summaries = {}
    for label, results_dir in zip(labels, results_dirs):
        try:
            summaries[label] = load_summary(results_dir)
        except FileNotFoundError as e:
            print(f"Warning: Skipping {label}: {e}")
    
    if not summaries:
        print("No valid results found")
        return
    
    all_metrics = set()
    for summary in summaries.values():
        all_metrics.update(summary.keys())
    
    if metrics:
        all_metrics = set(metrics) & all_metrics
    
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON")
    print("=" * 80)
    
    header = f"{'Metric':<25}"
    for label in labels:
        if label in summaries:
            header += f" {label[:15]:>15}"
    print(header)
    print("-" * 80)
    
    for metric in sorted(all_metrics):
        row = f"{metric:<25}"
        for label in labels:
            if label in summaries and metric in summaries[label]:
                mean = summaries[label][metric]["mean"]
                std = summaries[label][metric]["std"]
                row += f" {mean:>7.4f}±{std:<5.3f}"
            else:
                row += f" {'N/A':>15}"
        print(row)
    
    print("=" * 80)


def analyze_improvements(results: Dict) -> None:
    """Analyze improvement patterns in the results."""
    improvement_metrics = ["stoi_improvement", "si_snr_improvement"]
    
    for metric in improvement_metrics:
        values = []
        for wav_id, result in results.items():
            if isinstance(result, dict) and metric in result:
                val = result[metric]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    values.append((wav_id, val))
        
        if values:
            values.sort(key=lambda x: x[1])
            print(f"\n{metric} Analysis:")
            print(f"  Total samples: {len(values)}")
            print(f"  Improved: {sum(1 for _, v in values if v > 0)} "
                  f"({100*sum(1 for _, v in values if v > 0)/len(values):.1f}%)")
            print(f"  Degraded: {sum(1 for _, v in values if v < 0)} "
                  f"({100*sum(1 for _, v in values if v < 0)/len(values):.1f}%)")
            
            if len(values) >= 5:
                print(f"  Bottom 5:")
                for wav_id, val in values[:5]:
                    print(f"    {wav_id}: {val:.4f}")
                print(f"  Top 5:")
                for wav_id, val in values[-5:]:
                    print(f"    {wav_id}: {val:.4f}")


def plot_histograms(
    results: Dict,
    output_dir: str,
    metrics: Optional[List[str]] = None,
) -> None:
    """Plot histograms for each metric."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    metric_values = {}
    for wav_id, result in results.items():
        if isinstance(result, dict):
            for metric, value in result.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if metric not in metric_values:
                        metric_values[metric] = []
                    metric_values[metric].append(value)
    
    if metrics:
        metric_values = {k: v for k, v in metric_values.items() if k in metrics}
    
    for metric, values in metric_values.items():
        if len(values) < 10:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(values), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(values):.4f}')
        ax.axvline(np.median(values), color='g', linestyle='--', 
                   label=f'Median: {np.median(values):.4f}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {metric}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_hist.png'), dpi=150)
        plt.close()
    
    print(f"Plots saved to: {output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_evaluation(args) -> None:
    """Run the main evaluation pipeline."""
    # Load scp files
    logger.info(f"Loading enhanced audio from: {args.enh_scp}")
    enh_mapping = load_scp(args.enh_scp)
    logger.info(f"  Found {len(enh_mapping)} entries")
    
    logger.info(f"Loading reference audio from: {args.ref_scp}")
    ref_mapping = load_scp(args.ref_scp)
    logger.info(f"  Found {len(ref_mapping)} entries")
    
    mix_mapping = None
    if args.mix_scp:
        logger.info(f"Loading mixed audio from: {args.mix_scp}")
        mix_mapping = load_scp(args.mix_scp)
        logger.info(f"  Found {len(mix_mapping)} entries")
    
    # Load reference transcripts for WER
    ref_text_mapping = {}
    if "wer" in args.metrics:
        if args.ref_text:
            logger.info(f"Loading reference text from: {args.ref_text}")
            ref_text_mapping = load_text(args.ref_text)
            logger.info(f"  Found {len(ref_text_mapping)} entries")
        elif args.dialogue_jsonl:
            logger.info(f"Extracting transcripts from: {args.dialogue_jsonl}")
            ref_text_mapping = extract_transcripts_from_dialogue(args.dialogue_jsonl)
            logger.info(f"  Found {len(ref_text_mapping)} entries")
            
            # Save extracted transcripts for reference
            os.makedirs(args.output_dir, exist_ok=True)
            text_path = os.path.join(args.output_dir, "reference_text.txt")
            with open(text_path, 'w') as f:
                for wav_id in sorted(ref_text_mapping.keys()):
                    f.write(f"{wav_id} {ref_text_mapping[wav_id]}\n")
            logger.info(f"  Saved to: {text_path}")
    
    # Initialize ASR model if needed
    asr_model = None
    if "wer" in args.metrics:
        logger.info(f"Initializing Whisper ASR model: {args.whisper_model}")
        try:
             from versa import whisper_wer_setup
             asr_model = whisper_wer_setup(model_tag=args.whisper_model, use_gpu=(args.device == "cuda"))
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            args.metrics = [m for m in args.metrics if m != "wer"]

    # Initialize Versa models
    speaker_model = None
    discrete_predictors = None
    
    if "speaker" in args.metrics:
         logger.info("Initializing Speaker Model...")
         try:
             from versa import speaker_model_setup
             speaker_model = speaker_model_setup(use_gpu=(args.device == "cuda"))
         except Exception as e:
             logger.error(f"Failed to load Speaker model: {e}")
    
    if "discrete" in args.metrics:
         logger.info("Initializing Discrete Speech Predictors...")
         try:
             from versa import discrete_speech_setup
             discrete_predictors = discrete_speech_setup(use_gpu=(args.device == "cuda"))
         except Exception as e:
              logger.error(f"Failed to load Discrete predictors: {e}")
    
    # Find common keys
    common_keys = set(enh_mapping.keys()) & set(ref_mapping.keys())
    logger.info(f"Evaluating {len(common_keys)} common entries")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate each pair
    all_results = {}
    metrics_summary = {}
    
    from tqdm import tqdm
    for wav_id in tqdm(sorted(common_keys), desc="Evaluating"):
        enh_path = enh_mapping[wav_id]
        ref_path = ref_mapping[wav_id]
        mix_path = mix_mapping.get(wav_id) if mix_mapping else None
        reference_text = ref_text_mapping.get(wav_id)
        
        try:
            results = evaluate_pair(
                enh_path=enh_path,
                ref_path=ref_path,
                mix_path=mix_path,
                reference_text=reference_text,
                target_sr=args.target_sr,
                metrics=args.metrics,
                asr_model=asr_model,
                speaker_model=speaker_model,
                discrete_predictors=discrete_predictors,
            )
            all_results[wav_id] = results
            
            # Accumulate for summary (exclude string fields like hypothesis)
            for metric, value in results.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if metric not in metrics_summary:
                        metrics_summary[metric] = []
                    metrics_summary[metric].append(value)
                    
        except Exception as e:
            logger.error(f"Error evaluating {wav_id}: {e}")
            all_results[wav_id] = {"error": str(e)}
    
    # Compute summary statistics
    summary = {}
    for metric, values in metrics_summary.items():
        if values:
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }
    
    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved per-utterance results to: {results_path}")
    
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to: {summary_path}")
    
    # Save hypothesis transcriptions if WER was computed
    if "wer" in args.metrics:
        hyp_path = os.path.join(args.output_dir, "hypothesis.txt")
        with open(hyp_path, 'w') as f:
            for wav_id in sorted(all_results.keys()):
                if isinstance(all_results[wav_id], dict) and "hypothesis" in all_results[wav_id]:
                    f.write(f"{wav_id} {all_results[wav_id]['hypothesis']}\n")
        logger.info(f"Saved hypotheses to: {hyp_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for metric in sorted(summary.keys()):
        stats = summary[metric]
        print(f"{metric:20s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(min={stats['min']:.4f}, max={stats['max']:.4f}, n={stats['count']})")
    print("=" * 60)
    
    # Run improvement analysis
    if any(k.endswith("_improvement") for k in metrics_summary.keys()):
        analyze_improvements(all_results)


def run_comparison(args) -> None:
    """Run comparison of multiple result directories."""
    compare_checkpoints(args.results_dir, args.labels, args.compare_metrics)
    
    # Optionally analyze each directory
    if args.analyze:
        for i, results_dir in enumerate(args.results_dir):
            label = args.labels[i] if args.labels and i < len(args.labels) else results_dir
            print(f"\n{'='*60}")
            print(f"Analyzing: {label}")
            print(f"{'='*60}")
            
            try:
                results = load_results(results_dir)
                analyze_improvements(results)
                
                if args.plot:
                    output_dir = os.path.join(results_dir, "plots")
                    plot_histograms(results, output_dir)
            except Exception as e:
                print(f"Error analyzing {results_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate speech enhancement results with configurable metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Mode selection
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare mode: compare multiple result directories",
    )
    
    # Evaluation mode arguments
    eval_group = parser.add_argument_group("Evaluation Mode")
    eval_group.add_argument(
        "--enh_scp",
        type=str,
        help="Path to enhanced audio scp file",
    )
    eval_group.add_argument(
        "--ref_scp",
        type=str,
        help="Path to reference audio scp file",
    )
    eval_group.add_argument(
        "--mix_scp",
        type=str,
        default=None,
        help="Path to mixed/noisy audio scp file (for improvement metrics)",
    )
    eval_group.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results",
    )
    eval_group.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["stoi", "estoi", "si_snr", "sdr", "pesq", "dnsmos", "utmos", "wer", "speaker"],
        # default=["stoi", "mos", "wer", "speaker"],
        # default=["mos", "wer"],
        choices=ALL_METRICS,
        help=f"Metrics to compute. Available: {ALL_METRICS}",
    )
    eval_group.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Target sample rate for audio",
    )
    
    # WER-specific arguments
    wer_group = parser.add_argument_group("WER Options")
    wer_group.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Path to reference text file (format: wav_id text)",
    )
    wer_group.add_argument(
        "--dialogue_jsonl",
        type=str,
        default=None,
        help="Path to dialogues_all.jsonl (for transcript extraction)",
    )
    wer_group.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        help="Whisper model size (tiny, base, small, medium, large)",
    )
    wer_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for ASR (cuda or cpu)",
    )
    
    # Comparison mode arguments
    compare_group = parser.add_argument_group("Comparison Mode")
    compare_group.add_argument(
        "--results_dir",
        type=str,
        nargs="+",
        help="Result directories to compare",
    )
    compare_group.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each result directory",
    )
    compare_group.add_argument(
        "--compare_metrics",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to include in comparison",
    )
    compare_group.add_argument(
        "--analyze",
        action="store_true",
        help="Run improvement analysis on each directory",
    )
    compare_group.add_argument(
        "--plot",
        action="store_true",
        help="Generate histogram plots",
    )
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.results_dir:
            parser.error("--results_dir is required in compare mode")
        run_comparison(args)
    else:
        if not args.enh_scp or not args.ref_scp or not args.output_dir:
            parser.error("--enh_scp, --ref_scp, and --output_dir are required in evaluation mode")
        run_evaluation(args)


if __name__ == "__main__":
    main()
