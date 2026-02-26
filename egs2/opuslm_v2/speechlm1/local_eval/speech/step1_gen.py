import argparse
import json
import os
import random
import yaml
from collections import defaultdict
from pathlib import Path
import jinja2
import librosa
import soundfile as sf
import numpy as np
import re
import warnings

from local_split.sft_vllm_client import VLLMClient
from local_split.jsonl_parallel_runner import JsonlParallelRunner

from local_eval.speech.rir import apply_rir

random.seed(7)
MAX_DURATION = os.environ.get("MAX_DURATION", 15.0) # set max duration to 15s by default, can be configured by env var
MIN_DURATION = os.environ.get("MIN_DURATION", 5.0) # set min duration to 3s by default, can be configured by env var
warnings.filterwarnings("ignore")

def safe_load(path, sr=None):
    if not path or not os.path.exists(path):
        return None, None
    try:
        y, s = librosa.load(path, sr=sr)
        return y, s
    except Exception as e:
        return None, None

def save_audio(y, sr, path):
    assert y is not None, f"can not save None at {path}"
    sf.write(path, y, sr)

def apply_reverb(y, sr, size_str):
    if y is None: return None, size_str
    actual_size = "medium"
    if "small" in size_str: 
        actual_size = "small"
    elif "medium" in size_str: 
        actual_size = "medium"
    elif "large" in size_str: 
        actual_size = "large"
    
    y_rev = apply_rir(y, sr, actual_size)
    
    if y_rev.ndim == 1:
        res = y_rev[:len(y)]
    else:
        res = y_rev[:, :y.shape[1]]
    return res, actual_size

def apply_volume(y, sr, delta):
    import pyloudnorm as pyln
    if y.ndim == 1:
        y = y[np.newaxis, :]
    
    delta = 10 * max(delta / 5, 1) ** (1 / 4)
    duration = y.shape[1] / sr

    meter = pyln.Meter(
        sr, block_size=min(0.4, duration - 1e-10)
    )
    loudness = meter.integrated_loudness(y.T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loudness_normalized_audio = pyln.normalize.loudness(y.T, loudness, loudness + delta)

    return loudness_normalized_audio.T.squeeze()

def formulate_audio(y):
    y = np.nan_to_num(y)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y

def preprocess_and_sample(input_jsonl, output_jsonl, k):
    buckets = defaultdict(list)
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                duration = data['duration']
                text = data['text']
                if "utt_id" in data:
                    data["id"] = data.pop("utt_id") #<- re-name
                elif "id" in data:
                    pass
                elif "example_id" in data:
                    data["id"] = data.pop("example_id") #<- re-name
                else:
                    raise ValueError("No valid ID field found in data")

                if duration < 2.0 or len(text.split()) < 3:
                    continue
                if duration > MAX_DURATION or duration < MIN_DURATION:
                    continue

                buckets[int(duration)].append(line)
            except Exception:
                raise

    total_available = sum(len(b) for b in buckets.values())
    if total_available <= k:
        sampled_lines = [line for b in buckets.values() for line in b]
    else:
        sampled_lines = []
        available_buckets = {b: list(items) for b, items in buckets.items()}
        
        # Sort buckets to ensure deterministic sampling order
        sorted_buckets = sorted(available_buckets.keys())
        bucket_idx = 0
        
        while len(sampled_lines) < k and available_buckets:
            b = sorted_buckets[bucket_idx % len(sorted_buckets)]
            if b in available_buckets:
                item = random.choice(available_buckets[b])
                sampled_lines.append(item)
                available_buckets[b].remove(item)
                if not available_buckets[b]:
                    del available_buckets[b]
                    sorted_buckets.remove(b)
                    if not sorted_buckets:
                        break
                    # Adjust index since we removed an element
                    bucket_idx = bucket_idx % len(sorted_buckets)
                    continue
            bucket_idx += 1
                
    print(f"Sampled {len(sampled_lines)} lines from {total_available} available lines.")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line)

def get_captioner_ref(captioner_client, audio_path):
    """Helper to query the Multimodal Captioner to generate a reference caption."""
    caption_resp = captioner_client.chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{audio_path}"}
                }
            ]
        }],
    )
    return caption_resp

def judge_edit(judge_client, params, original_text, original_caption, new_text, new_caption, edit_operation, **kwargs):
    """Helper to invoke the judge LLM to validate the edit operation."""

    sys_prompt = params['judge_system_prompt']
    user_prompt = jinja2.Template(params['judge_user_prompt']).render(
        original_text=original_text,
        original_caption=original_caption,
        new_text=new_text,
        new_caption=new_caption,
        edit_operation=edit_operation,
        **kwargs
    )

    try:
        resp = judge_client.chat_completion(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
            json_mode=True,
        )
        if isinstance(resp, dict):
            return resp['valid'], resp['reason']
    except Exception as e:
        print(f"Judge error: {e}")
    return False, "Judge evaluation failed or returned invalid format"

def process_transcription(idx, line, cat_name, params, llm_client, judge_client, captioner_client, output_dir):
    data = json.loads(line)

    system_prompt = params['system_prompt']
    user_prompt_tmpl = jinja2.Template(params['user_prompt'])
    
    render_kwargs = dict(
        text=data["text"],
        audio_caption=data["audio_caption"]
    )
    
    if cat_name == "transcription_ins":
        min_w, max_w = params['words_range']
        render_kwargs.update({'min_w': min_w, 'max_w': max_w})
        model_param = f"ins_{min_w}_{max_w}"
    elif cat_name == "transcription_del":
        min_w, max_w = params['words_range']
        render_kwargs.update({'min_w': min_w, 'max_w': max_w})
        model_param = f"del_{min_w}_{max_w}"
    elif cat_name == "transcription_sub":
        min_w, max_w = params['words_range']
        render_kwargs.update({'min_w': min_w, 'max_w': max_w})
        model_param = f"sub_{min_w}_{max_w}"
    elif cat_name == "transcription_replace_sentence":
        model_param = "replace_all"
    elif cat_name == "transcription_add_paralinguistic":
        tags = params['tags']
        tag = tags[idx % len(tags)]
        render_kwargs.update({'tag': tag})
        model_param = tag
    else:
        return None

    user_prompt = user_prompt_tmpl.render(**render_kwargs)
    
    try:
        resp = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            json_mode=True,
        )
        # print(data["text"])
        # print(resp['edit_operation'])
        # print(resp['new_asr_text'])

        data['edit_type'] = cat_name
        data['edit_prompt'] = resp['edit_operation'] # Moved outside edit_kwargs
        data['edit_kwargs'] = {'param': model_param} # Only pure params left

        data['target_text'] = resp['new_asr_text']
        data['target_audio_caption'] = resp['new_caption']
        data['target_audio_path'] = None

        # Judge Content
        is_valid, reason = judge_edit(
            judge_client, params,
            data['text'], data['audio_caption'],
            data['target_text'], data['target_audio_caption'],
            data['edit_prompt'] # Using edit_prompt directly
        )
        if not is_valid:
            # print(f"{data['text']=}", f"{data['target_text']=}", f"{data['edit_prompt']=}", reason)
            return None
        data['judge_reason'] = reason

        return data
    except Exception as e:
        print(f"LLM error: {e}")
    return None

def process_style(idx, line, cat_name, params, llm_client, judge_client, captioner_client, output_dir):
    data = json.loads(line)
    
    system_prompt = params['system_prompt']
    user_prompt_tmpl = jinja2.Template(params['user_prompt'])
    
    if cat_name == "style_whisper":
        styles = params['styles']
        style = styles[idx % len(styles)]
    elif cat_name == "style_emotion":
        emotions = params['emotions']
        style = emotions[idx % len(emotions)]
    else:
        return None
        
    user_prompt = user_prompt_tmpl.render(style=style, **data)
    
    try:
        resp = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            json_mode=True,
        )

        data['edit_type'] = cat_name
        data['edit_kwargs'] = {'style': style} # Only pure params left
        data['edit_prompt'] = resp['edit_prompt'] # Moved outside edit_kwargs
        
        data['target_text'] = data["text"]
        data['target_audio_caption'] = resp['new_caption']
        data['target_audio_path'] = None

        # Judge Content
        is_valid, reason = judge_edit(
            judge_client, params,
            data['text'], data['audio_caption'],
            data['target_text'], data['target_audio_caption'],
            data['edit_prompt'], # Using edit_prompt directly
            style=style
        )
        if not is_valid:
            return None
        data['judge_reason'] = reason

        return data
    except Exception as e:
        print(f"LLM error: {e}")
    return None

def process_audio_effect(idx, line, cat_name, params, llm_client, judge_client, captioner_client, output_dir):
    data = json.loads(line)
    audio_path = data['audio_path']

    if not audio_path or not os.path.exists(audio_path):
        return None
        
    SR = 16000
    y, sr = safe_load(audio_path, SR)
    if y is None:
        return None
        
    base_name = Path(audio_path).stem
    target_audio_path = os.path.join(output_dir, f"{idx:05d}_{base_name}_{cat_name}.wav")
    
    edit_kwargs = {}
    
    if cat_name == "audio_effect_speed":
        param_val = random.choice(params['range'])
        y_out = librosa.effects.time_stretch(y, rate=param_val)
        edit_kwargs = {'speed_rate': param_val}
        
        data['edit_prompt'] = f"Please recognize the language of this speech and transcribe it. And adjusts the speed to {param_val}." # Generate prompt locally
    elif cat_name == "audio_effect_volume":
        param_val = random.choice(params['range'])
        y_out = apply_volume(y, SR, param_val)
        edit_kwargs = {'volume_gain': param_val}
        
        data['edit_prompt'] = f"Please recognize the language of this speech and transcribe it. And adjusts the volume to {param_val}." # Generate prompt locally
    elif cat_name == "audio_effect_pitch":
        param_val = random.choice(params['range'])
        y_out = librosa.effects.pitch_shift(y, sr=SR, n_steps=param_val)
        edit_kwargs = {'pitch_steps': param_val}
        
        data['edit_prompt'] = f"Please recognize the language of this speech and transcribe it. And shifts the pitch by {param_val} steps." # Generate prompt locally
    elif cat_name == "audio_effect_reverb":
        types = params['types']
        param_val = types[idx % len(types)]
        y_out, actual_size = apply_reverb(y, SR, param_val)
        if y_out is None:
            return None
        edit_kwargs = {'reverb_size': actual_size}

        data['edit_prompt'] = f"Please recognize the language of this speech and transcribe it. And make the speech sound like in a {param_val} room." # Generate prompt locally
    elif cat_name == "audio_effect_dereverb":
        types = params['types']
        param_val = types[idx % len(types)]
        y_reverb, actual_size = apply_reverb(y, SR, param_val)
        if y_reverb is None:
            return None
        edit_kwargs = {'dereverb_from_size': actual_size}
            
        target_audio_path = os.path.join(output_dir, f"{idx:05d}_{base_name}_dereverb_source.wav")
        save_audio(formulate_audio(y_reverb), SR, target_audio_path)
        
        data['audio_path'] = os.path.abspath(target_audio_path)
        data['target_audio_path'] = os.path.abspath(audio_path)
        data['target_text'] = data['text']
        data['edit_prompt'] = f"Please recognize the language of this speech and transcribe it. And denoise the audio." # Generate prompt locally
        data['needs_recaption'] = True
    else:
        return None
    # "Please recognize the language of this speech and transcribe it. And add rain to audio.\n", TODO

    # Use LLM-client to update caption based on audio effect
    sys_prompt = params['system_prompt']
    user_prompt_tmpl = jinja2.Template(params['user_prompt'])
    user_prompt = user_prompt_tmpl.render(
        audio_caption=data["audio_caption"],
        effect_param=edit_kwargs
    )
    resp = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        json_mode=True,
    )
    target_audio_caption= resp['new_caption']

    data['edit_type'] = cat_name
    data['edit_kwargs'] = edit_kwargs

    if not cat_name == "audio_effect_dereverb":
        y_out = formulate_audio(y_out)
        save_audio(y_out, SR, target_audio_path)
        data['target_audio_path'] = os.path.abspath(target_audio_path)
        data['target_text'] = data['text']
        # Use Captioner Client to generate reference caption on target audio
        data['target_audio_caption_ref'] = get_captioner_ref(captioner_client, data['target_audio_path'])
        data['target_audio_caption'] = target_audio_caption # gen by llm
    else:
        data['target_audio_caption'] = data.pop("audio_caption")
        data['audio_caption'] = target_audio_caption
        data['audio_caption_ref'] = get_captioner_ref(captioner_client, data['audio_path'])

    # Judge Edit Validation
    is_valid, reason = judge_edit(
        judge_client, params,
        data['text'], data['audio_caption'],
        data['target_text'], data['target_audio_caption'],
        data['edit_prompt'] # Pass generated prompt
    )
    
    if not is_valid:
        return None
    data['judge_reason'] = reason

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_jsonl", required=True, help="Input metadata.jsonl")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    parser.add_argument("-c", "--config", required=True, help="Path to config.yaml")
    parser.add_argument("-k", "--k", type=int, default=None, help="Number of samples per category")
    parser.add_argument("--nj", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    wav_dir = os.path.join(args.output_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    
    vllm_cfg = config.pop('vllm')
    llm_client = VLLMClient(
        base_url=vllm_cfg.pop('url', "http://localhost:8001/v1"),
        model=vllm_cfg.pop('model', "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"),
        max_concurrent=args.nj * 2,
        timeout=1200,
        default_kwargs=vllm_cfg
    )

    judge_cfg = config.pop('judge')
    judge_client = VLLMClient(
        base_url=judge_cfg.pop('url', "http://localhost:8001/v1"),
        model=judge_cfg.pop('model', "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"),
        max_concurrent=args.nj * 2,
        timeout=1200,
        default_kwargs=judge_cfg
    )

    captioner_cfg = config.pop('captioner')
    captioner_client = VLLMClient(
        base_url=captioner_cfg.pop('url', "http://localhost:9000/v1"),
        model=captioner_cfg.pop('model', "Qwen/Qwen3-Omni-30B-A3B-Captioner"),
        max_concurrent=args.nj * 2,
        timeout=1200,
        default_kwargs=captioner_cfg
    )

    for cat_name, params in config.items():
        print(f"Processing {cat_name}...")
        output_jsonl = os.path.join(args.output_dir, f"{cat_name}.jsonl")
        sampled_input = os.path.join(args.output_dir, f"tmp_sampled_{cat_name}.jsonl")
        preprocess_and_sample(args.input_jsonl, sampled_input, args.k)

        def _process(idx, line, current_cat=cat_name, current_params=params):
            try:
                if current_cat.startswith('audio_effect_'):
                    return process_audio_effect(idx, line, current_cat, current_params, llm_client=llm_client, judge_client=judge_client, captioner_client=captioner_client, output_dir=wav_dir)
                elif current_cat.startswith('style_'):
                    return process_style(idx, line, current_cat, current_params, llm_client=llm_client, judge_client=judge_client, captioner_client=captioner_client, output_dir=wav_dir)
                elif current_cat.startswith('transcription_'):
                    return process_transcription(idx, line, current_cat, current_params,  llm_client=llm_client, judge_client=judge_client, captioner_client=captioner_client, output_dir=wav_dir)
            except Exception as e:
                print(f"Error processing {current_cat} line {idx}: {e}")
                return None

        runner = JsonlParallelRunner(
            input_jsonl=sampled_input,
            output_jsonl=output_jsonl,
            process_fn=_process,
            n_jobs=args.nj,
            desc=f"{cat_name}",
            resume=False
        )
        runner.run()
        
        if os.path.exists(sampled_input):
            os.remove(sampled_input)

if __name__ == "__main__":
    main()
