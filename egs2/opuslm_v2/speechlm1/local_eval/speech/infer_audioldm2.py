"""
AudioLDM2 batch inference script.

Pipeline:
  raw caption  ──► LLM (vLLM / OpenAI-compatible)  ──► ≤2-sentence caption
  ≤2-sentence caption  ──► AudioLDM2  ──► wav

Single-GPU execution; set CUDA_VISIBLE_DEVICES externally.
Output format mirrors infer_cv3_light.py:
  <output_dir>/<file_stem>/<utt_id>.wav   (audio files)
  <output_dir>/<file_stem>.scp            (kaldi-style scp)
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = (
    "You are a concise audio caption editor. "
    "Rewrite the given audio caption into at most two sentences. "
    "Preserve all essential acoustic information: sound events, acoustic environment, "
    "temporal structure, and perceptual qualities (e.g., timbre, loudness, rhythm). "
    "Remove repetition and redundant detail. "
    "Output ONLY the rewritten caption, with no more than 20 words — no quotes, no prefix, no explanation."
)

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def compress_caption(caption: str, client: OpenAI, model: str) -> str:
    """Compress an audio caption to ≤2 sentences via the LLM."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user",   "content": caption},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def compress_captions_parallel(
    captions: List[str],
    client: OpenAI,
    model: str,
    max_workers: int = 16,
) -> List[str]:
    """Compress a list of captions concurrently (I/O-bound, thread pool)."""
    results: List[Optional[str]] = [None] * len(captions)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(compress_caption, cap, client, model): i
            for i, cap in enumerate(captions)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                tqdm.write(f"[LLM Error] caption[{idx}]: {exc}")
                results[idx] = captions[idx]  # fallback to raw
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# AudioLDM2 generation helper
# ---------------------------------------------------------------------------

def generate_audio_batch(
    pipe: AudioLDM2Pipeline,
    prompts: List[str],
    num_inference_steps: int,
    audio_length_in_s: float,
    guidance_scale: float,
) -> list:
    """Run AudioLDM2 on a batch of prompts; returns one float32 waveform per prompt."""
    result = pipe(
        prompts,
        num_inference_steps=num_inference_steps,
        audio_length_in_s=audio_length_in_s,
        guidance_scale=guidance_scale,
        num_waveforms_per_prompt=1,
    )
    # result.audios shape: (B, T)
    audios = result.audios if hasattr(result, "audios") else result[0]  # type: ignore[union-attr]
    return [audios[i] for i in range(len(prompts))]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="AudioLDM2 inference with LLM caption compression"
    )

    # I/O
    parser.add_argument(
        "--jsonl-files", type=str, nargs="+", required=True,
        help="Input JSONL file(s). Each line must have 'id' and a caption field.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Root output directory.",
    )
    parser.add_argument(
        "--caption-field", type=str, default="caption",
        help="JSONL field that holds the raw caption text (default: 'caption').",
    )

    # AudioLDM2
    parser.add_argument(
        "--model-id", type=str, default="cvssp/audioldm2-large",
        help="HuggingFace repo ID or local path for AudioLDM2.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=200,
        help="Diffusion steps (default: 200).",
    )
    parser.add_argument(
        "--audio-length-in-s", type=float, default=10.0,
        help="Generated audio length in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=3.5,
        help="Classifier-free guidance scale (default: 3.5).",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Output wav sample rate (default: 16000).",
    )

    # LLM (vLLM OpenAI-compatible server)
    parser.add_argument(
        "--llm-base-url", type=str, default="http://cnode1-001:8000/v1",
        help="Base URL of the vLLM OpenAI-compatible server.",
    )
    parser.add_argument(
        "--llm-api-key", type=str, default="EMPTY",
        help="API key placeholder (any non-empty string works for vLLM).",
    )
    parser.add_argument(
        "--llm-model", type=str, default="auto",
        help=(
            "Model name as registered in vLLM. "
            "Use 'auto' to query the server and pick the first available model."
        ),
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Bypass LLM compression and feed the raw caption directly to AudioLDM2.",
    )

    # Batching
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of prompts per GPU forward pass (default: 8). "
             "Increase for higher GPU utilisation; reduce if OOM.",
    )
    parser.add_argument(
        "--llm-workers", type=int, default=16,
        help="Parallel threads for LLM caption compression (default: 16).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_jsonl():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load AudioLDM2 ──────────────────────────────────────────────────────
    print(f"[Info] Loading AudioLDM2 from '{args.model_id}' with bfloat16 …")
    pipe = AudioLDM2Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    print("[Info] AudioLDM2 model loaded on CUDA.\n")

    # ── LLM client ─────────────────────────────────────────────────────────
    llm_client: OpenAI | None = None
    llm_model = args.llm_model

    if not args.skip_llm:
        llm_client = OpenAI(base_url=args.llm_base_url, api_key=args.llm_api_key)
        if llm_model == "auto":
            models = llm_client.models.list()
            if not models.data:
                raise RuntimeError(
                    f"No models found at {args.llm_base_url}. "
                    "Start vLLM first or use --skip-llm."
                )
            llm_model = models.data[0].id
            print(f"[Info] LLM model auto-detected: {llm_model}")
        else:
            print(f"[Info] LLM model: {llm_model}")
        print(f"[Info] LLM endpoint: {args.llm_base_url}\n")
    else:
        print("[Info] --skip-llm: raw captions will be used directly.\n")

    # ── Per-file loop ──────────────────────────────────────────────────────
    for jsonl_file in args.jsonl_files:
        if not os.path.exists(jsonl_file):
            print(f"[Warning] File not found: {jsonl_file} — skipping.")
            continue

        file_stem = Path(jsonl_file).stem
        save_dir  = os.path.join(args.output_dir, file_stem)
        scp_path  = os.path.join(args.output_dir, f"{file_stem}.scp")
        os.makedirs(save_dir, exist_ok=True)

        print(f"--- Loading {jsonl_file} ---")
        tasks = []
        with open(jsonl_file, "r", encoding="utf-8") as fin:
            for line_idx, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[Error] JSONDecodeError at line {line_idx} — skipped.")

        # ── Separate already-done tasks ────────────────────────────────
        pending, done_count = [], 0
        for task in tasks:
            utt_id = task["id"]
            save_audio_path = os.path.abspath(os.path.join(save_dir, f"{utt_id}.wav"))
            if os.path.exists(save_audio_path):
                done_count += 1
            else:
                pending.append(task)

        success_count = done_count
        if done_count:
            print(f"[Info] {done_count}/{len(tasks)} already generated — skipping.")

        with open(scp_path, "w", encoding="utf-8") as fscp:
            # Write already-done entries first
            for task in tasks:
                utt_id = task["id"]
                save_audio_path = os.path.abspath(os.path.join(save_dir, f"{utt_id}.wav"))
                if os.path.exists(save_audio_path):
                    fscp.write(f"{utt_id}\t{save_audio_path}\n")

            # ── Batched inference over pending tasks ──────────────────────
            bs = args.batch_size
            pbar = tqdm(total=len(pending), desc=f"Processing {file_stem}")

            for batch_start in range(0, len(pending), bs):
                batch = pending[batch_start : batch_start + bs]
                utt_ids    = [t["id"] for t in batch]
                raw_caps   = [
                    t.get("target_audio_caption") or t.get("target_caption") or t.get("text", "")
                    for t in batch
                ]
                save_paths = [
                    os.path.abspath(os.path.join(save_dir, f"{uid}.wav"))
                    for uid in utt_ids
                ]

                try:
                    # ── Step 1: LLM compression (parallel threads) ────────
                    if llm_client is not None:
                        compressed_caps = compress_captions_parallel(
                            raw_caps, llm_client, llm_model,
                            max_workers=args.llm_workers,
                        )
                        for uid, raw, comp in zip(utt_ids, raw_caps, compressed_caps):
                            tqdm.write(
                                f"[LLM] {uid}\n"
                                f"  raw : {raw}\n"
                                f"  comp: {comp}"
                            )
                    else:
                        compressed_caps = raw_caps

                    # ── Step 2: AudioLDM2 batch generation ────────────────
                    audio_list = generate_audio_batch(
                        pipe,
                        prompts=compressed_caps,
                        num_inference_steps=args.num_inference_steps,
                        audio_length_in_s=args.audio_length_in_s,
                        guidance_scale=args.guidance_scale,
                    )

                    # ── Step 3: Save each wav ─────────────────────────────
                    for uid, audio_np, save_path in zip(utt_ids, audio_list, save_paths):
                        sf.write(save_path, audio_np, samplerate=args.sample_rate)
                        fscp.write(f"{uid}\t{save_path}\n")
                        success_count += 1
                        tqdm.write(f"[Save] {save_path}")

                except Exception as exc:
                    tqdm.write(f"[Failed] batch {utt_ids}: {exc}")

                pbar.update(len(batch))
                pbar.set_postfix_str(f"Success: {success_count}/{len(tasks)}")

            pbar.close()

        print(f"--- Finished {file_stem} ({success_count}/{len(tasks)} succeeded) ---")
        print(f"  Wavs → {save_dir}")
        print(f"  SCP  → {scp_path}\n")


if __name__ == "__main__":
    process_jsonl()

# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
# 1. Start vLLM server (separate terminal):
#    vllm serve Qwen/Qwen3-4B --port 8080
#
# 2. Run inference:
#    CUDA_VISIBLE_DEVICES=0 python infer_audioldm2.py \
#        --jsonl-files /path/to/test.jsonl \
#        --output-dir  /path/to/output \
#        --caption-field caption \
#        --llm-base-url http://localhost:8080/v1 \
#        --llm-model auto \
#        --num-inference-steps 200 \
#        --audio-length-in-s 10.0 \
#        --guidance-scale 3.5 \
#        --batch-size 8 \
#        --llm-workers 16
#
# Input JSONL format (one JSON object per line):
#   {"id": "utt_0001", "caption": "A dog barking in a park followed by children laughing."}
#
# Output:
#   <output_dir>/<file_stem>/utt_0001.wav
#   <output_dir>/<file_stem>.scp   (tab-separated: utt_id  /abs/path/to/utt_0001.wav)
