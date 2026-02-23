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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from local_split.sft_vllm_client import VLLMClient
from local_split.jsonl_parallel_runner import JsonlParallelRunner

try:
    from pair_generation import apply_rir
except ImportError:
    try:
        from rir import apply_rir
    except ImportError:
        pass

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
                duration = data.get('duration', 0)
                text = data.get('text', '')
                if duration < 3.0 or len(text.split()) < 3:
                    continue
                buckets[int(duration)].append(line)
            except Exception:
                continue
    
    total_available = sum(len(b) for b in buckets.values())
    if total_available <= k:
        sampled_lines = [line for b in buckets.values() for line in b]
    else:
        sampled_lines = []
        available_buckets = {b: list(items) for b, items in buckets.items()}
        while len(sampled_lines) < k and available_buckets:
            b = random.choice(list(available_buckets.keys()))
            item = random.choice(available_buckets[b])
            sampled_lines.append(item)
            available_buckets[b].remove(item)
            if not available_buckets[b]:
                del available_buckets[b]
                
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line)

def process_transcription(idx, line, cat_name, params, llm_client, output_dir):
    data = json.loads(line)

    system_prompt = params.get('system_prompt', '')
    user_prompt_tmpl = jinja2.Template(params.get('user_prompt', ''))
    
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
        tag = random.choice(tags)
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
            temperature=0.7,
            max_tokens=1024,
            json_mode=True,
        )
        if isinstance(resp, dict):
            data['target_text'] = resp['new_asr_text']
            data['target_audio_caption'] = resp['new_caption']
            data['edit_type'] = cat_name
            data['edit_kwargs'] = {'operation': resp['edit_operation'], 'param': model_param}
            data['target_audio_path'] = ""
            return data
    except Exception as e:
        print(f"LLM error: {e}")
    return None

def process_style(idx, line, cat_name, params, llm_client, output_dir):
    data = json.loads(line)
    
    system_prompt = params.get('system_prompt', '')
    user_prompt_tmpl = jinja2.Template(params.get('user_prompt', ''))
    
    if cat_name == "style_whisper":
        style = random.choice(params['styles'])
    elif cat_name == "style_emotion":
        style = random.choice(params['emotions'])
    else:
        return None
        
    user_prompt = user_prompt_tmpl.render(text=data["text"], style=style)
    
    try:
        resp = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
            json_mode=True,
        )
        if isinstance(resp, dict):
            data['target_text'] = resp['modified_text']
            data['target_audio_caption'] = data['audio_caption']
            data['edit_type'] = cat_name
            data['edit_kwargs'] = {'instruction': resp['style_instruction'], 'style': style}
            data['target_audio_path'] = ""
            return data
    except Exception as e:
        print(f"LLM error: {e}")
    return None

def process_audio_effect(idx, line, cat_name, params, output_dir):
    data = json.loads(line)
    audio_path = data.get('audio_path', '')

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
        val = random.uniform(params['range'][0], params['range'][1])
        param_val = round(val, 2)
        y_out = librosa.effects.time_stretch(y, rate=param_val)
        edit_kwargs = {'speed_rate': param_val}
        
    elif cat_name == "audio_effect_volume":
        val = random.uniform(params['range'][0], params['range'][1])
        param_val = round(val, 2)
        y_out = apply_volume(y, SR, param_val)
        edit_kwargs = {'volume_gain': param_val}
        
    elif cat_name == "audio_effect_pitch":
        val = random.uniform(params['range'][0], params['range'][1])
        param_val = round(val, 2)
        y_out = librosa.effects.pitch_shift(y, sr=SR, n_steps=param_val)
        edit_kwargs = {'pitch_steps': param_val}
        
    elif cat_name == "audio_effect_reverb":
        param_val = random.choice(params['types'])
        y_out, actual_size = apply_reverb(y, SR, param_val)
        if y_out is None:
            return None
        edit_kwargs = {'reverb_size': actual_size}
        
    elif cat_name == "audio_effect_dereverb":
        param_val = random.choice(params['types'])
        y_reverb, actual_size = apply_reverb(y, SR, param_val)
        if y_reverb is None:
            return None
            
        # For dereverb, we swap source and target
        # The original audio becomes the target (clean), and the reverberated audio becomes the source
        target_audio_path = os.path.join(output_dir, f"{idx:05d}_{base_name}_dereverb_source.wav")
        save_audio(formulate_audio(y_reverb), SR, target_audio_path)
        
        data['audio_path'] = os.path.abspath(target_audio_path)
        data['target_audio_path'] = os.path.abspath(audio_path)
        data['target_text'] = data['text']
        data['target_audio_caption'] = data["audio_caption"]
        data['edit_type'] = cat_name
        data['edit_kwargs'] = {'dereverb_from_size': actual_size}
        data['needs_recaption'] = True # Flag for later recaptioning
        return data
    else:
        return None
        
    y_out = formulate_audio(y_out)
    save_audio(y_out, SR, target_audio_path)
    
    data['target_audio_path'] = os.path.abspath(target_audio_path)
    data['target_text'] = data['text']
    data['target_audio_caption'] = data['audio_caption']
    data['edit_type'] = cat_name
    data['edit_kwargs'] = edit_kwargs
    
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
        
    k = args.k if args.k is not None else config.get('sampling', {}).get('k', 10)
    
    os.makedirs(args.output_dir, exist_ok=True)
    wav_dir = os.path.join(args.output_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    
    vllm_cfg = config.get('vllm', {})
    llm_client = VLLMClient(
        base_url=vllm_cfg.get('url', "http://localhost:8001/v1"),
        model=vllm_cfg.get('model', "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"),
        max_concurrent=args.nj * 2,
        timeout=1200,
    )

    for cat_name, params in config.items():
        if cat_name in ['vllm', 'sampling']:
            continue
            
        print(f"Processing {cat_name}...")
        
        # Sample data for this category
        sampled_input = os.path.join(args.output_dir, f"tmp_sampled_{cat_name}.jsonl")
        preprocess_and_sample(args.input_jsonl, sampled_input, k)
        
        output_jsonl = os.path.join(args.output_dir, f"{cat_name}.jsonl")
        
        def _process(idx, line, current_cat=cat_name, current_params=params):
            if current_cat.startswith('audio_effect_'):
                return process_audio_effect(idx, line, current_cat, current_params, wav_dir)
            elif current_cat.startswith('style_'):
                return process_style(idx, line, current_cat, current_params, llm_client, wav_dir)
            elif current_cat.startswith('transcription_'):
                return process_transcription(idx, line, current_cat, current_params, llm_client, wav_dir)
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
        
        # Clean up temporary sampled file
        if os.path.exists(sampled_input):
            os.remove(sampled_input)

if __name__ == "__main__":
    main()
