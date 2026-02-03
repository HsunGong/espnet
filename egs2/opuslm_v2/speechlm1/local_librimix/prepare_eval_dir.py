#!/usr/bin/env python3
"""
Prepare evaluation directory for librimix enhancement evaluation.

This script parses inference results and original dialogues to create:
1. enh.scp: wav.scp for enhanced speech (wav-id enh-wav-path)
2. ref.scp: wav.scp for reference clean speech (wav-id ref-wav-path)  
3. mix.scp: wav.scp for mixed noisy speech (wav-id mix-wav-path)

Input1: results.json <- parse enh wav-path + dataset-id (new id)
Input2: dialogue.jsonl <- parse dataset-id + original wav-path (the basename is wav-id)
Output: enh.scp, ref.scp, mix.scp

Example paths:
- results.json: exp/opuslm_v2_stage3_sft_librimix_enh-v3/inference/inference_step_351881/dialogue_librimix_spk1_enh-v3_test100/results.json
- dialogue.jsonl: /mnt/home/xungong-andr-1766e0/prep/data/librimix_sft/data_mix_single/test100/spk1_enh/a2a_enh-v3/stage3_dialogues/dialogues_all.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_results_json(results_path: str) -> Dict[str, str]:
    """
    Parse results.json to extract dataset-id -> enhanced wav path mapping.
    
    The results.json format is:
    {
        "dataset-id": [
            ["assistant", "text", "..."],
            ["assistant", "audio", "path/to/enhanced.wav"]
        ],
        ...
    }
    
    Returns:
        Dict mapping dataset-id (example_id) to enhanced wav path
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get the directory of results.json for resolving relative paths
    results_dir = Path(results_path).parent
    # Find the root directory (speechlm1/)
    root_dir = results_dir
    while root_dir.name != 'speechlm1' and root_dir.parent != root_dir:
        root_dir = root_dir.parent
    
    enh_mapping = {}
    for dataset_id, messages in results.items():
        # Find the audio message from assistant
        for msg in messages:
            if len(msg) >= 3 and msg[0] == "assistant" and msg[1] == "audio":
                audio_path = msg[2]
                # Handle relative paths - they're relative to speechlm1/
                if not os.path.isabs(audio_path):
                    audio_path = str(root_dir / audio_path)
                enh_mapping[dataset_id] = audio_path
                break
    
    return enh_mapping


def parse_dialogue_jsonl(dialogue_path: str) -> Dict[str, dict]:
    """
    Parse dialogues_all.jsonl to extract example_id -> metadata mapping.
    
    The dialogue format includes:
    {
        "example_id": "librimix-enhancement_0_keep_request",
        "messages": [...],
        "metadata": {
            "idx": "1235-135884-0016_8014-280382-0001",  # wav-id
            "mix_audio_path": "...",
            "source_audio_paths": ["ref_path", "noise_path"],
            ...
        }
    }
    
    Returns:
        Dict mapping example_id to metadata dict containing wav-id, ref_path, mix_path
    """
    dialogue_mapping = {}
    
    with open(dialogue_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            example_id = data.get("example_id", "")
            metadata = data.get("metadata", {})
            
            # Extract wav-id from metadata
            wav_id = metadata.get("idx", "")
            mix_audio_path = metadata.get("mix_audio_path", "")
            source_audio_paths = metadata.get("source_audio_paths", [])
            
            # source_audio_paths[0] is typically the clean speech (s1)
            # source_audio_paths[1] is typically the noise
            ref_audio_path = source_audio_paths[0] if source_audio_paths else ""
            
            dialogue_mapping[example_id] = {
                "wav_id": wav_id,
                "mix_audio_path": mix_audio_path,
                "ref_audio_path": ref_audio_path,
            }
    
    return dialogue_mapping


def prepare_eval_dir(
    results_path: str,
    dialogue_path: str,
    output_dir: str,
    check_exists: bool = True,
) -> Tuple[int, int]:
    """
    Prepare evaluation directory with scp files.
    
    Args:
        results_path: Path to results.json from inference
        dialogue_path: Path to dialogues_all.jsonl
        output_dir: Output directory for scp files
        check_exists: Whether to check if audio files exist
        
    Returns:
        Tuple of (success_count, total_count)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse input files
    print(f"Parsing results from: {results_path}")
    enh_mapping = parse_results_json(results_path)
    print(f"  Found {len(enh_mapping)} enhanced audio entries")
    
    print(f"Parsing dialogues from: {dialogue_path}")
    dialogue_mapping = parse_dialogue_jsonl(dialogue_path)
    print(f"  Found {len(dialogue_mapping)} dialogue entries")
    
    # Match and create scp files
    enh_scp_path = os.path.join(output_dir, "enh.scp")
    ref_scp_path = os.path.join(output_dir, "ref.scp")
    mix_scp_path = os.path.join(output_dir, "mix.scp")
    
    success_count = 0
    total_count = 0
    missing_enh = []
    missing_ref = []
    
    with open(enh_scp_path, 'w') as f_enh, \
         open(ref_scp_path, 'w') as f_ref, \
         open(mix_scp_path, 'w') as f_mix:
        
        for example_id, enh_path in sorted(enh_mapping.items()):
            total_count += 1
            
            if example_id not in dialogue_mapping:
                print(f"Warning: example_id '{example_id}' not found in dialogue")
                continue
            
            meta = dialogue_mapping[example_id]
            wav_id = meta["wav_id"]
            ref_path = meta["ref_audio_path"]
            mix_path = meta["mix_audio_path"]
            
            if not wav_id:
                print(f"Warning: No wav_id for example_id '{example_id}'")
                continue
            
            # Check if files exist
            if check_exists:
                if not os.path.exists(enh_path):
                    missing_enh.append(enh_path)
                    continue
                if ref_path and not os.path.exists(ref_path):
                    missing_ref.append(ref_path)
                    continue
            
            # Write to scp files
            f_enh.write(f"{wav_id} {enh_path}\n")
            if ref_path:
                f_ref.write(f"{wav_id} {ref_path}\n")
            if mix_path:
                f_mix.write(f"{wav_id} {mix_path}\n")
            
            success_count += 1
    
    # Report missing files
    if missing_enh:
        print(f"Warning: {len(missing_enh)} enhanced audio files not found")
        if len(missing_enh) <= 5:
            for p in missing_enh:
                print(f"  {p}")
    if missing_ref:
        print(f"Warning: {len(missing_ref)} reference audio files not found")
        if len(missing_ref) <= 5:
            for p in missing_ref:
                print(f"  {p}")
    
    print(f"\nOutput files created in: {output_dir}")
    print(f"  - enh.scp: {success_count} entries")
    print(f"  - ref.scp: {success_count} entries")
    print(f"  - mix.scp: {success_count} entries")
    print(f"Success rate: {success_count}/{total_count}")
    
    return success_count, total_count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare evaluation directory for librimix enhancement evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_json",
        type=str,
        required=True,
        help="Path to results.json from inference output",
    )
    parser.add_argument(
        "--dialogue_jsonl",
        type=str,
        required=True,
        help="Path to dialogues_all.jsonl from SFT data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for scp files",
    )
    parser.add_argument(
        "--no_check_exists",
        action="store_true",
        help="Skip checking if audio files exist",
    )
    
    args = parser.parse_args()
    
    prepare_eval_dir(
        results_path=args.results_json,
        dialogue_path=args.dialogue_jsonl,
        output_dir=args.output_dir,
        check_exists=not args.no_check_exists,
    )


if __name__ == "__main__":
    main()
