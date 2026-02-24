import argparse
import json
import os
import random
import uuid
import warnings

import jinja2
import librosa
import numpy as np
import soundfile as sf
import yaml

from local_split.sft_vllm_client import VLLMClient
from local_split.jsonl_parallel_runner import JsonlParallelRunner

random.seed(7)
warnings.filterwarnings("ignore")

SR = 16000


# =========================
# Audio helpers
# =========================

def safe_load(path, sr=SR):
    if not path or not os.path.exists(path):
        return None
    try:
        y, _ = librosa.load(path, sr=sr)
        return y
    except Exception:
        return None


def save_audio(y, sr, path):
    if y is None:
        raise ValueError(f"can not save None at {path}")
    sf.write(path, y, sr)


def formulate_audio(y):
    y = np.nan_to_num(y)
    m = np.max(np.abs(y))
    if m > 0:
        y = y / m
    return y


def mix_audio(y1, y2):
    # y1 is main; y2 is added with auto RMS scaling, max 0.8 effective gain
    if len(y1) > len(y2):
        out2 = np.pad(y2, (0, len(y1) - len(y2)))
    else:
        out2 = y2[: len(y1)]

    y1_rms = float(np.mean(np.abs(y1)) + 1e-8)
    y2_rms = float(np.mean(np.abs(out2)) + 1e-8)
    scale = y2_rms / y1_rms

    if scale > 0.8:
        mixed = y1 + 0.8 * out2 / scale
    else:
        mixed = y1 + out2

    return formulate_audio(mixed)


# =========================
# Data helpers
# =========================

def filter_data(jsonl_path):
    out = []
    if not os.path.exists(jsonl_path):
        return out

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["audio_caption"] = d.get("audio_caption", d.pop("qwen_caption", None))
            dur = d.get("duration", None)
            if dur is None or not d["audio_caption"]:
                continue
            if 8.0 <= float(dur) <= 12.0:
                out.append(d)
    return out


def generate_mix_tasks(pools, output_task_jsonl, k, target_main_type, op):
    if not pools.get(target_main_type):
        return False

    def pick_delta(delta_type, main_data):
        main_dur = float(main_data["duration"])
        if delta_type == "speech":
            cands = [x for x in pools["speech"] if float(x["duration"]) <= main_dur]
            if not cands:
                return None
            return random.choice(cands)
        return random.choice(pools[delta_type])

    tasks = []
    for _ in range(k):
        main_data = random.choice(pools[target_main_type])
        if target_main_type == "speech":
            delta_type = random.choice(["music", "sound"])
        else:
            delta_type = random.choice(["speech", "music", "sound"])

        if op == "ADD":
            y_data = pick_delta(delta_type, main_data)
            if y_data is None:
                continue
            tasks.append({
                "operation": "ADD",
                "main_type": target_main_type,
                "delta_type": delta_type,
                "edit_type": f"{target_main_type}_add_{delta_type}",
                "main_data": main_data,
                "y_data": y_data,
            })

        elif op == "REMOVE":
            x_data = pick_delta(delta_type, main_data)
            if x_data is None:
                continue
            tasks.append({
                "operation": "REMOVE",
                "main_type": target_main_type,
                "delta_type": delta_type,
                "edit_type": f"{target_main_type}_remove_{delta_type}",
                "main_data": main_data,
                "x_data": x_data,
            })

        elif op == "REPLACE":
            x_data = pick_delta(delta_type, main_data)
            y_data = pick_delta(delta_type, main_data)
            if x_data is None or y_data is None:
                continue
            if x_data["audio_path"] == y_data["audio_path"]:
                continue
            tasks.append({
                "operation": "REPLACE",
                "main_type": target_main_type,
                "delta_type": delta_type,
                "edit_type": f"{target_main_type}_replace_{delta_type}",
                "main_data": main_data,
                "x_data": x_data,
                "y_data": y_data,
            })

        else:
            return False

    if not tasks:
        return False

    with open(output_task_jsonl, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return True


# =========================
# LLM helpers
# =========================

def get_captioner_ref(captioner_client, audio_path):
    try:
        resp = captioner_client.chat_completion(
            messages=[{
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}}],
            }],
        )
        return resp
    except Exception:
        return ""


def judge_edit(judge_client, judge_system_prompt, judge_user_prompt, **kwargs):
    user_prompt = jinja2.Template(judge_user_prompt).render(**kwargs)
    resp = judge_client.chat_completion(
        messages=[
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        json_mode=True,
    )
    # 不用 .get：缺 key 直接让上层返回 None
    return resp["valid"], resp["reason"]


# ============================================================
# 3 independent operation functions
# ============================================================

def run_add_op(
    idx,
    task,
    system_prompt,
    user_prompt,
    judge_system_prompt,
    judge_user_prompt,
    llm_client,
    judge_client,
    captioner_client,
    output_dir,
):
    """
    Prompt contract (ADD): llm_resp must contain exactly:
      - edit_prompt
      - target_caption
    """
    try:
        main_data = task["main_data"]
        y_data = task["y_data"]

        main_path = main_data["audio_path"]
        main_caption = main_data["audio_caption"]

        y_main = safe_load(main_path, SR)
        if y_main is None:
            return None

        base = f'{task["main_type"]}_add_{idx:05d}'
        input_wav = os.path.abspath(os.path.join(output_dir, f"{base}_input.wav"))
        target_wav = os.path.abspath(os.path.join(output_dir, f"{base}_target.wav"))

        # 1) LLM
        user_render = jinja2.Template(user_prompt).render(
            source_caption=main_caption,
            add_caption=y_data["audio_caption"],
        )
        llm_resp = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_render},
            ],
            json_mode=True,
        )

        edit_prompt = llm_resp["edit_prompt"]
        target_caption = llm_resp["target_caption"]

        # 2) Audio
        y_Y = safe_load(y_data["audio_path"], SR)
        if y_Y is None:
            return None

        y_input = y_main
        y_target = mix_audio(y_main, y_Y)
        save_audio(y_input, SR, input_wav)
        save_audio(y_target, SR, target_wav)

        # 3) Caption refs (source ref 直接复用 main_caption)
        source_caption_ref = main_caption
        target_caption_ref = get_captioner_ref(captioner_client, target_wav)

        # 4) Judge
        is_valid, reason = judge_edit(
            judge_client,
            judge_system_prompt=judge_system_prompt,
            judge_user_prompt=judge_user_prompt,
            source_caption=main_caption,                 # 绝对事实
            source_caption_ref=source_caption_ref,       # “unreliable”但这里复用 main
            target_caption=target_caption,
            target_caption_ref=target_caption_ref,
            edit_prompt=edit_prompt,
            add_caption_ref=y_data["audio_caption"],     # delta ingredient
        )
        if not is_valid:
            return None

        # 5) Final record
        return {
            "id": uuid.uuid4().hex,
            "metadata": {
                "main":task["main_data"],
                "y_data": task["y_data"],
                "operation": "add",
            },
            "audio_path": input_wav,
            "audio_caption": main_caption,  # ADD 不生成 source_caption，输入就是 main
            "audio_caption_ref": source_caption_ref,
            "target_audio_path": target_wav,
            "target_caption": target_caption,
            "target_caption_ref": target_caption_ref,
            "main_type": task["main_type"],
            "delta_type": task["delta_type"],
            "edit_type": task["edit_type"],
            "edit_prompt": edit_prompt,
            "judge_reason": reason,
            "add_audio_path": os.path.abspath(y_data["audio_path"]),
            "add_audio_caption": y_data["audio_caption"],
        }

    except Exception:
        # 不要 .get；任何异常/缺 key -> None
        return None


def run_remove_op(
    idx,
    task,
    system_prompt,
    user_prompt,
    judge_system_prompt,
    judge_user_prompt,
    llm_client,
    judge_client,
    captioner_client,
    output_dir,
):
    """
    Prompt contract (REMOVE): llm_resp must contain exactly:
      - edit_prompt
      - source_caption
    """
    try:
        main_data = task["main_data"]
        x_data = task["x_data"]

        main_path = main_data["audio_path"]
        main_caption = main_data["audio_caption"]

        y_main = safe_load(main_path, SR)
        if y_main is None:
            return None

        base = f'{task["main_type"]}_remove_{idx:05d}'
        input_wav = os.path.abspath(os.path.join(output_dir, f"{base}_input.wav"))
        target_wav = os.path.abspath(os.path.join(output_dir, f"{base}_target.wav"))

        # 1) LLM
        user_render = jinja2.Template(user_prompt).render(
            target_caption=main_caption,
            remove_caption=x_data["audio_caption"],
        )
        llm_resp = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_render},
            ],
            json_mode=True,
        )

        edit_prompt = llm_resp["edit_prompt"]
        source_caption = llm_resp["source_caption"]

        # 2) Audio
        y_X = safe_load(x_data["audio_path"], SR)
        if y_X is None:
            return None

        y_input = mix_audio(y_main, y_X)  # mixed (source)
        y_target = y_main                 # clean (target)
        save_audio(y_input, SR, input_wav)
        save_audio(y_target, SR, target_wav)

        # 3) Caption refs (target ref 复用 main_caption)
        source_caption_ref = get_captioner_ref(captioner_client, input_wav)
        target_caption_ref = main_caption

        # 4) Judge
        is_valid, reason = judge_edit(
            judge_client,
            judge_system_prompt=judge_system_prompt,
            judge_user_prompt=judge_user_prompt,
            source_caption=source_caption,
            source_caption_ref=source_caption_ref,
            target_caption=main_caption,                 # 绝对事实
            target_caption_ref=target_caption_ref,
            edit_prompt=edit_prompt,
            remove_caption_ref=x_data["audio_caption"],  # delta ingredient
        )
        if not is_valid:
            return None

        # 5) Final record
        return {
            "id": uuid.uuid4().hex,
            "metadata": {
                "main":task["main_data"],
                "x_data": task["x_data"],
                "operation": "remove",
            },
            "audio_path": input_wav,
            "audio_caption": source_caption,
            "audio_caption_ref": source_caption_ref,
            "target_audio_path": target_wav,
            "target_caption": main_caption,  # REMOVE 不生成 target_caption，target 就是 main
            "target_caption_ref": target_caption_ref,
            "main_type": task["main_type"],
            "delta_type": task["delta_type"],
            "edit_type": task["edit_type"],
            "edit_prompt": edit_prompt,
            "judge_reason": reason,
            "remove_audio_path": os.path.abspath(x_data["audio_path"]),
            "remove_audio_caption": x_data["audio_caption"],
        }

    except Exception:
        return None


def run_replace_op(
    idx,
    task,
    system_prompt,
    user_prompt,
    judge_system_prompt,
    judge_user_prompt,
    llm_client,
    judge_client,
    captioner_client,
    output_dir,
):
    """
    Prompt contract (REPLACE): llm_resp must contain exactly:
      - edit_prompt
      - source_caption
      - target_caption
    """
    try:
        main_data = task["main_data"]
        x_data = task["x_data"]
        y_data = task["y_data"]

        main_path = main_data["audio_path"]
        main_caption = main_data["audio_caption"]

        y_main = safe_load(main_path, SR)
        if y_main is None:
            return None

        base = f'{task["main_type"]}_replace_{idx:05d}'
        input_wav = os.path.abspath(os.path.join(output_dir, f"{base}_input.wav"))
        target_wav = os.path.abspath(os.path.join(output_dir, f"{base}_target.wav"))

        # 1) LLM
        user_render = jinja2.Template(user_prompt).render(
            core_caption=main_caption,
            remove_caption=x_data["audio_caption"],
            add_caption=y_data["audio_caption"],
        )
        llm_resp = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_render},
            ],
            json_mode=True,
        )

        edit_prompt = llm_resp["edit_prompt"]
        source_caption = llm_resp["source_caption"]
        target_caption = llm_resp["target_caption"]

        # 2) Audio
        y_X = safe_load(x_data["audio_path"], SR)
        y_Y = safe_load(y_data["audio_path"], SR)
        if y_X is None or y_Y is None:
            return None

        y_input = mix_audio(y_main, y_X)
        y_target = mix_audio(y_main, y_Y)
        save_audio(y_input, SR, input_wav)
        save_audio(y_target, SR, target_wav)

        # 3) Caption refs (两个都打标)
        source_caption_ref = get_captioner_ref(captioner_client, input_wav)
        target_caption_ref = get_captioner_ref(captioner_client, target_wav)

        # 4) Judge
        is_valid, reason = judge_edit(
            judge_client,
            judge_system_prompt=judge_system_prompt,
            judge_user_prompt=judge_user_prompt,
            source_caption=source_caption,
            source_caption_ref=source_caption_ref,
            target_caption=target_caption,
            target_caption_ref=target_caption_ref,
            edit_prompt=edit_prompt,
            original_core_ref=main_caption,
            original_remove_ref=x_data["audio_caption"],
            original_add_ref=y_data["audio_caption"],
        )
        if not is_valid:
            return None

        # 5) Final record
        return {
            "id": uuid.uuid4().hex,
            "metadata": {
                "main":task["main_data"],
                "x_data": task["x_data"],
                "y_data": task["y_data"],
                "operation": "replace",
            },
            "audio_path": input_wav,
            "audio_caption": source_caption,
            "audio_caption_ref": source_caption_ref,
            "target_audio_path": target_wav,
            "target_audio_caption": target_caption,
            "target_audio_caption_ref": target_caption_ref,
            "main_type": task["main_type"],
            "delta_type": task["delta_type"],
            "edit_type": task["edit_type"],
            "edit_prompt": edit_prompt,
            "judge_reason": reason,
            "add_audio_path": os.path.abspath(y_data["audio_path"]),
            "add_audio_caption": y_data["audio_caption"],
            "remove_audio_path": os.path.abspath(x_data["audio_path"]),
            "remove_audio_caption": x_data["audio_caption"],
        }

    except Exception:
        return None



# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech", required=True, help="Input metadata.speech.jsonl")
    parser.add_argument("--music", required=True, help="Input metadata.music.jsonl")
    parser.add_argument("--sound", required=True, help="Input metadata.sound.jsonl")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    parser.add_argument("-c", "--config", required=True, help="Path to config.yaml")
    parser.add_argument("-k", "--k", type=int, default=100, help="Number of samples per (main_type, op)")
    parser.add_argument("--nj", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    wav_dir = os.path.join(args.output_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    vllm_cfg = config["vllm"]
    judge_cfg = config["judge"]
    captioner_cfg = config["captioner"]

    llm_client = VLLMClient(
        base_url=vllm_cfg["url"],
        model=vllm_cfg["model"],
        max_concurrent=args.nj * 2,
    )
    judge_client = VLLMClient(
        base_url=judge_cfg["url"],
        model=judge_cfg["model"],
        max_concurrent=args.nj * 2,
    )
    captioner_client = VLLMClient(
        base_url=captioner_cfg["url"],
        model=captioner_cfg["model"],
        max_concurrent=args.nj * 2,
    )

    print("Step 1: Filtering data (8~12s)...")
    pools = {
        "speech": filter_data(args.speech),
        "music": filter_data(args.music),
        "sound": filter_data(args.sound),
    }
    for t, arr in pools.items():
        print(f"  {t}: {len(arr)} valid samples")

    types = ["speech", "music", "sound"]
    ops = ["ADD", "REMOVE", "REPLACE"]

    for main_type in types:
        if not pools[main_type]:
            continue

        for op in ops:
            print(f"\n================ {main_type.upper()} | {op} ================")

            tmp_tasks = os.path.join(args.output_dir, f"tmp_mix_tasks_{main_type}_{op.lower()}.jsonl")
            out_jsonl = os.path.join(args.output_dir, f"{main_type}_{op.lower()}_mix.jsonl")

            if not generate_mix_tasks(pools, tmp_tasks, args.k, target_main_type=main_type, op=op):
                print("  Skipped (No tasks generated)")
                continue

            op_params = config[f"{op.lower()}_params"]

            def _process(idx, line):
                try:
                    task = json.loads(line)
                    op = task["operation"]

                    if op == "ADD":
                        return run_add_op(
                            idx, task,
                            system_prompt=op_params["system_prompt"],
                            user_prompt=op_params["user_prompt"],
                            judge_system_prompt=op_params["judge_system_prompt"],
                            judge_user_prompt=op_params["judge_user_prompt"],
                            llm_client=llm_client,
                            judge_client=judge_client,
                            captioner_client=captioner_client,
                            output_dir=wav_dir,
                        )

                    if op == "REMOVE":
                        return run_remove_op(
                            idx, task,
                            system_prompt=op_params["system_prompt"],
                            user_prompt=op_params["user_prompt"],
                            judge_system_prompt=op_params["judge_system_prompt"],
                            judge_user_prompt=op_params["judge_user_prompt"],
                            llm_client=llm_client,
                            judge_client=judge_client,
                            captioner_client=captioner_client,
                            output_dir=wav_dir,
                        )

                    if op == "REPLACE":
                        return run_replace_op(
                            idx, task,
                            system_prompt=op_params["system_prompt"],
                            user_prompt=op_params["user_prompt"],
                            judge_system_prompt=op_params["judge_system_prompt"],
                            judge_user_prompt=op_params["judge_user_prompt"],
                            llm_client=llm_client,
                            judge_client=judge_client,
                            captioner_client=captioner_client,
                            output_dir=wav_dir,
                        )

                    return None

                except Exception:
                    return None

            runner = JsonlParallelRunner(
                input_jsonl=tmp_tasks,
                output_jsonl=out_jsonl,
                process_fn=_process,
                backend="loky",
                n_jobs=args.nj,
                desc=f"{main_type}_{op}",
                resume=False,
            )
            runner.run()

            if os.path.exists(tmp_tasks):
                os.remove(tmp_tasks)

    print("\nGeneration complete for all main types and operations!")


if __name__ == "__main__":
    main()
