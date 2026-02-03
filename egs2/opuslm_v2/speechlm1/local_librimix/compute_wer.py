#!/usr/bin/env python3
"""
Compute Word Error Rate (WER) for enhanced audio using ASR models.

This script provides WER evaluation using:
- Whisper (OpenAI)
- OWSM (if available)
- ESPnet ASR (if available)

Usage:
    python compute_wer.py --enh_scp enh.scp --ref_text ref.txt --output_dir results/
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
    """Load audio file and resample if necessary."""
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


class WhisperASR:
    """Whisper ASR wrapper."""
    
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        import whisper
        self.model = whisper.load_model(model_name, device=device)
        self.device = device
        
    def transcribe(self, audio: np.ndarray) -> str:
        result = self.model.transcribe(audio.astype(np.float32))
        return result["text"].strip()


class OWSMasR:
    """OWSM ASR wrapper (if available)."""
    
    def __init__(self, model_tag: str = "default", device: str = "cuda"):
        try:
            from espnet2.bin.s2t_inference import Speech2Text
            self.model = Speech2Text.from_pretrained(
                model_tag=model_tag,
                device=device,
            )
            self.available = True
        except Exception as e:
            logger.warning(f"OWSM not available: {e}")
            self.available = False
    
    def transcribe(self, audio: np.ndarray) -> str:
        if not self.available:
            return ""
        result = self.model(audio)
        return result[0][0] if result else ""


def normalize_text(text: str) -> str:
    """Normalize text for WER computation."""
    import re
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def compute_wer(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute WER and CER between reference and hypothesis."""
    import jiwer
    
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    
    if not ref_norm:
        return {"wer": float('nan'), "cer": float('nan')}
    
    wer = jiwer.wer(ref_norm, hyp_norm)
    cer = jiwer.cer(ref_norm, hyp_norm)
    
    return {
        "wer": float(wer),
        "cer": float(cer),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute WER for enhanced audio using ASR models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--enh_scp",
        type=str,
        required=True,
        help="Path to enhanced audio scp file",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Path to reference text file (format: wav_id text)",
    )
    parser.add_argument(
        "--ref_scp",
        type=str,
        default=None,
        help="Path to reference audio scp (for ASR-based reference)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--asr_model",
        type=str,
        default="whisper",
        choices=["whisper", "owsm"],
        help="ASR model to use",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        help="Whisper model size (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    
    args = parser.parse_args()
    
    # Load audio files
    logger.info(f"Loading enhanced audio from: {args.enh_scp}")
    enh_mapping = load_scp(args.enh_scp)
    logger.info(f"  Found {len(enh_mapping)} entries")
    
    # Load reference text if provided
    ref_text_mapping = {}
    if args.ref_text:
        logger.info(f"Loading reference text from: {args.ref_text}")
        ref_text_mapping = load_text(args.ref_text)
        logger.info(f"  Found {len(ref_text_mapping)} entries")
    
    # Load reference audio if provided (for ASR-based reference)
    ref_audio_mapping = {}
    if args.ref_scp:
        logger.info(f"Loading reference audio from: {args.ref_scp}")
        ref_audio_mapping = load_scp(args.ref_scp)
        logger.info(f"  Found {len(ref_audio_mapping)} entries")
    
    # Initialize ASR model
    logger.info(f"Initializing {args.asr_model} ASR model...")
    if args.asr_model == "whisper":
        asr = WhisperASR(model_name=args.whisper_model, device=args.device)
    elif args.asr_model == "owsm":
        asr = OWSMasR(device=args.device)
    else:
        raise ValueError(f"Unknown ASR model: {args.asr_model}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each audio
    all_results = {}
    wer_values = []
    cer_values = []
    
    from tqdm import tqdm
    for wav_id in tqdm(sorted(enh_mapping.keys()), desc="Computing WER"):
        enh_path = enh_mapping[wav_id]
        
        try:
            # Load enhanced audio and transcribe
            enh_audio, sr = load_audio(enh_path)
            enh_hyp = asr.transcribe(enh_audio)
            
            result: Dict[str, str] = {"enh_hypothesis": enh_hyp}
            
            # Get reference text
            ref_text = ""
            if wav_id in ref_text_mapping:
                ref_text = ref_text_mapping[wav_id]
            elif wav_id in ref_audio_mapping:
                # Transcribe reference audio
                ref_audio, _ = load_audio(ref_audio_mapping[wav_id])
                ref_text = asr.transcribe(ref_audio)
                result["ref_hypothesis"] = ref_text
            
            if ref_text:
                result["reference"] = ref_text
                wer_result = compute_wer(ref_text, enh_hyp)
                result["wer"] = str(wer_result["wer"])
                result["cer"] = str(wer_result["cer"])
                
                if not np.isnan(wer_result["wer"]):
                    wer_values.append(wer_result["wer"])
                if not np.isnan(wer_result["cer"]):
                    cer_values.append(wer_result["cer"])
            
            all_results[wav_id] = result
            
        except Exception as e:
            logger.error(f"Error processing {wav_id}: {e}")
            all_results[wav_id] = {"error": str(e)}
    
    # Compute summary
    summary = {}
    if wer_values:
        summary["wer"] = {
            "mean": float(np.mean(wer_values)),
            "std": float(np.std(wer_values)),
            "min": float(np.min(wer_values)),
            "max": float(np.max(wer_values)),
            "count": len(wer_values),
        }
    if cer_values:
        summary["cer"] = {
            "mean": float(np.mean(cer_values)),
            "std": float(np.std(cer_values)),
            "min": float(np.min(cer_values)),
            "max": float(np.max(cer_values)),
            "count": len(cer_values),
        }
    
    # Save results
    results_path = os.path.join(args.output_dir, "wer_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved per-utterance results to: {results_path}")
    
    summary_path = os.path.join(args.output_dir, "wer_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to: {summary_path}")
    
    # Save transcriptions
    hyp_path = os.path.join(args.output_dir, "hypothesis.txt")
    with open(hyp_path, 'w') as f:
        for wav_id in sorted(all_results.keys()):
            if "enh_hypothesis" in all_results[wav_id]:
                f.write(f"{wav_id} {all_results[wav_id]['enh_hypothesis']}\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("WER EVALUATION SUMMARY")
    print("=" * 60)
    for metric in sorted(summary.keys()):
        stats = summary[metric]
        print(f"{metric.upper():10s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(min={stats['min']:.4f}, max={stats['max']:.4f}, n={stats['count']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
