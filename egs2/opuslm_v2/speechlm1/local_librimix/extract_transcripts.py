#!/usr/bin/env python3
"""
Extract transcripts from librimix dialogue data.

This script extracts source text transcriptions from the dialogue jsonl files
to create reference text files for WER evaluation.

Usage:
    python extract_transcripts.py --dialogue_jsonl path/to/dialogues_all.jsonl --output_dir output/
"""

import argparse
import json
import logging
import os
import re
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for consistency."""
    # Convert to uppercase (librispeech convention)
    text = text.upper()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def extract_transcripts(dialogue_path: str) -> Dict[str, str]:
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
                # Clean up the text
                text = source_texts[0]
                text = normalize_text(text)
                transcripts[wav_id] = text
    
    return transcripts


def main():
    parser = argparse.ArgumentParser(
        description="Extract transcripts from librimix dialogue data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dialogue_jsonl",
        type=str,
        required=True,
        help="Path to dialogues_all.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for transcript files",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="kaldi",
        choices=["kaldi", "json", "tsv"],
        help="Output format: kaldi (wav_id text), json, or tsv",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Extracting transcripts from: {args.dialogue_jsonl}")
    transcripts = extract_transcripts(args.dialogue_jsonl)
    logger.info(f"Found {len(transcripts)} transcripts")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.format == "kaldi":
        # Kaldi text format: wav_id text
        output_path = os.path.join(args.output_dir, "text")
        with open(output_path, 'w') as f:
            for wav_id in sorted(transcripts.keys()):
                f.write(f"{wav_id} {transcripts[wav_id]}\n")
    
    elif args.format == "json":
        output_path = os.path.join(args.output_dir, "transcripts.json")
        with open(output_path, 'w') as f:
            json.dump(transcripts, f, indent=2)
    
    elif args.format == "tsv":
        output_path = os.path.join(args.output_dir, "transcripts.tsv")
        with open(output_path, 'w') as f:
            f.write("wav_id\ttext\n")
            for wav_id in sorted(transcripts.keys()):
                f.write(f"{wav_id}\t{transcripts[wav_id]}\n")
    
    logger.info(f"Saved transcripts to: {output_path}")
    
    # Print some statistics
    total_words = sum(len(t.split()) for t in transcripts.values())
    avg_words = total_words / len(transcripts) if transcripts else 0
    print(f"\nStatistics:")
    print(f"  Total utterances: {len(transcripts)}")
    print(f"  Total words: {total_words}")
    print(f"  Average words per utterance: {avg_words:.1f}")


if __name__ == "__main__":
    main()
