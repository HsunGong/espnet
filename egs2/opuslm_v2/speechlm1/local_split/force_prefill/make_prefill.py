#!/usr/bin/env python3
import argparse
import copy
import json
import os
from pathlib import Path
import soundfile as sf
import random
random.seed(7)

def get_audio_duration(file_path):
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception as e:
        # Fallback to reading whole file if info fails or for robustness
        try:
            f = sf.SoundFile(file_path)
            return len(f) / f.samplerate
        except Exception as e2:
            print(f"Error reading audio {file_path}: {e2}")
            return 0.0

def find_assistant_audio_message(messages):
    if not isinstance(messages, list):
        return -1, None
    for idx, msg in enumerate(messages):
        if isinstance(msg, list) and len(msg) >= 3 and msg[0] == "assistant" and msg[1] == "audio":
            return idx, msg[2]
        if isinstance(msg, list) and len(msg) >= 3 and msg[1] == "audio":
            return idx, msg[2]
    return -1, None


def split_and_save_audio(input_audio_path, output_audio_path, target_duration):
    try:
        audio, sr = sf.read(input_audio_path)
        if sr <= 0:
            return 0.0
        max_frames = len(audio)
        target_frames = int(target_duration * sr)
        if target_frames <= 0:
            return 0.0
        target_frames = min(target_frames, max_frames)
        clipped = audio[:target_frames]
        sf.write(output_audio_path, clipped, sr)
        return float(target_frames) / float(sr)
    except Exception as e:
        print(f"Error splitting audio {input_audio_path}: {e}")
        return 0.0


def process_file_content(lines, limit, step_sec, min_dur, max_dur, wavs_dir):
    dialogue_results = []
    lines_to_process = random.choices(lines, k=limit) if limit > 0 else lines
    
    for i, line in enumerate(lines_to_process):
        try:
            data = json.loads(line)
            
            messages = data.get("messages", [])
            audio_msg_idx, audio_path = find_assistant_audio_message(messages)
            
            if audio_msg_idx < 0 or not audio_path:
                continue
                
            if not os.path.exists(audio_path):
                continue
                
            duration = get_audio_duration(audio_path)
            if duration <= 0:
                continue
                
            max_split_dur = min(duration - 1, max_dur)
            
            base_example_id = data.get("example_id", f"sample-{i}")
            
            current_dur = min_dur
            while current_dur <= max_split_dur:
                new_sample = copy.deepcopy(data)
                current_dur_rounded = round(current_dur, 1)

                sample_id = f"{base_example_id}_dur{current_dur_rounded:.1f}s"
                safe_sample_id = (
                    sample_id.replace("/", "_")
                    .replace("\\", "_")
                    .replace(" ", "_")
                )
                split_audio_path = (wavs_dir / f"{safe_sample_id}.wav").resolve()

                saved_dur = split_and_save_audio(audio_path, str(split_audio_path), current_dur_rounded)
                if saved_dur <= 0:
                    current_dur += step_sec
                    continue

                new_sample["example_id"] = sample_id
                if isinstance(new_sample.get("messages"), list) and len(new_sample["messages"]) > audio_msg_idx:
                    new_sample["messages"][audio_msg_idx][2] = str(split_audio_path)

                dialogue_results.append(new_sample)
                
                current_dur += step_sec
                
        except json.JSONDecodeError:
            continue
    return dialogue_results

def main():
    parser = argparse.ArgumentParser(description="Generate split-duration dialogue samples from metadata.")
    parser.add_argument("-i", "--input_jsonls", action='append', required=True, help="Input jsonl file (can be used multiple times)")
    parser.add_argument("-o", "--output_jsonl", type=Path, required=True, help="Output jsonl file")
    parser.add_argument("-k", "--limit", type=int, default=-1, help="Limit number of samples per input file (-1 for all)")
    parser.add_argument("--step_sec", type=float, default=1, help="Step size in seconds for audio splitting")
    parser.add_argument("--min_dur", type=float, default=1, help="Minimum split duration in seconds")
    parser.add_argument("--max_dur", type=float, default=1e10, help="Maximum split duration in seconds")

    args = parser.parse_args()

    all_dialogue_results = []

    wavs_dir = args.output_jsonl.with_suffix("").absolute().resolve()
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    for input_file in args.input_jsonls:
        if os.path.exists(input_file):
            print(f"Processing {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            file_dialogue_results = process_file_content(
                lines, args.limit, args.step_sec, args.min_dur, args.max_dur, wavs_dir
            )
            all_dialogue_results.extend(file_dialogue_results)
        else:
            print(f"Input file not found: {input_file}")

    # Write dialogue JSONL
    print(f"Writing {len(all_dialogue_results)} dialogue samples to {args.output_jsonl}")
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for item in all_dialogue_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Write dataset JSON (jsonl + json)
    output_json = args.output_jsonl.with_suffix(".json")

    samples_ids = [item["example_id"] for item in all_dialogue_results]
    
    dataset_json = {
        "data_entry": [
            {
                "name": "dialogue",
                "path": os.path.abspath(args.output_jsonl),
                "reader": "dialogue", 
            }
        ],
        "samples": samples_ids,
    }

    print(f"Written dataset JSON to {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f_json:
        json.dump(dataset_json, f_json, ensure_ascii=False, indent=2)
        f_json.write("\n")


if __name__ == "__main__":
    main()

# python3 local_split/make_prefill.py -i data/mmau/music.t2a.jsonl -i data/mmau/speech.t2a.jsonl -i data/mmau/sound.t2a.jsonl -o data/debug.jsonl -k 2 --step 25
