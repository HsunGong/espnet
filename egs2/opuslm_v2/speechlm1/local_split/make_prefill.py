#!/usr/bin/env python3
import argparse
import json
import os
import sys
import soundfile as sf
from tqdm import tqdm

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

def process_file_content(lines, limit, step, min_len):
    results = []
    lines_to_process = lines[:limit] if limit > 0 else lines
    
    for i, line in enumerate(lines_to_process):
        try:
            data = json.loads(line)
            
            # Find audio path in messages
            audio_path = None
            if "messages" in data:
                for msg in data["messages"]:
                    if isinstance(msg, list) and len(msg) >= 3 and msg[1] == "audio":
                        audio_path = msg[2]
                        break
                    # Handle assistant audio message if format is different or standard
                    if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "assistant" and "audio" in msg: 
                        # This part depends on structure, assuming list [role, type, content]
                        pass
            
            if not audio_path:
                continue
                
            if not os.path.exists(audio_path):
                continue
                
            duration = get_audio_duration(audio_path)
            if duration <= 0:
                continue
                
            max_prefill_len = int(duration * 50)
            
            base_example_id = data.get("example_id", f"sample-{i}")
            
            current_len = min_len
            while current_len <= max_prefill_len:
                new_sample = data.copy()
                new_sample["prefill_len"] = current_len
                # Create a unique example_id for this sub-sample
                new_sample["example_id"] = f"{base_example_id}_prefill{current_len}"
                results.append(new_sample)
                
                current_len += step
                
        except json.JSONDecodeError:
            continue
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate prefill samples from metadata.")
    parser.add_argument("-i", "--input_jsonls", action='append', required=True, help="Input jsonl file (can be used multiple times)")
    parser.add_argument("-o", "--output_jsonl", required=True, help="Output jsonl file")
    parser.add_argument("-k", "--limit", type=int, default=-1, help="Limit number of samples per input file (-1 for all)")
    parser.add_argument("--step", type=int, default=25, help="Step size for prefill length generation")
    parser.add_argument("--min_len", type=int, default=25, help="Minimum prefill length")

    args = parser.parse_args()

    all_results = []
    
    for input_file in args.input_jsonls:
        if os.path.exists(input_file):
            print(f"Processing {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            file_results = process_file_content(lines, args.limit, args.step, args.min_len)
            all_results.extend(file_results)
        else:
            print(f"Input file not found: {input_file}")

    print(f"Writing {len(all_results)} samples to {args.output_jsonl}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    
    # Write JSONL
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Write Dataset JSON
    # Derive json path from jsonl path by changing suffix
    if args.output_jsonl.endswith('.jsonl'):
        output_json = args.output_jsonl[:-1] # .jsonl -> .json
    else:
        output_json = args.output_jsonl + '.json'
        
    samples_ids = [item["example_id"] for item in all_results]
    
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

    with open(output_json, 'w', encoding='utf-8') as f_json:
        json.dump(dataset_json, f_json, ensure_ascii=False, indent=2)
        f_json.write("\n")

    print(f"Written dataset JSON to {output_json}")

if __name__ == "__main__":
    main()
