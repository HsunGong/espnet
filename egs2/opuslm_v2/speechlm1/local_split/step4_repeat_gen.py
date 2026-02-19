#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys
from pathlib import Path
import logging

import numpy as np
import soundfile as sf

from local_split.local_config import apply_step_config
from local_split.jsonl_parallel_runner import JsonlParallelRunner

SILENCE = 0.

def build_repeat1_path(split1_audio_path: str, silence: float = 0.) -> str:
    """Convert `/path/to/foo.wav` -> `/path/to/foo.repeat1.wav`."""
    p = Path(split1_audio_path)
    if p.suffix:
        return str(p.with_suffix(f".repeat1.sil{silence:.1f}{p.suffix}"))
    return str(p.with_name(f"{p.name}.repeat1.sil{silence:.1f}.flac"))


def repeat_split1_audio(split1_audio_path: str, silence: float = 0.) -> tuple[str, float]:
    """Repeat split1 audio 2 times and save to `<stem>.repeat1.wav`.

    Returns:
        (new_audio_path, duration_seconds)
    """
    audio, sr = sf.read(split1_audio_path)

    # with some silence?
    silence_audio = np.zeros(int(sr * silence))

    # axis=0 repeats on time dimension for both mono and multi-channel audio
    repeated = np.concatenate([audio, silence_audio, audio], axis=0)

    out_path = build_repeat1_path(split1_audio_path, silence=silence)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sf.write(out_path, repeated, sr)
    duration = float(len(repeated) / sr) if sr > 0 else 0.0
    return out_path, duration


def process_one(idx: int, line: str) -> dict | None:
    try:
        record = json.loads(line)

        split1_data = record.get("split1", {})
        split1_audio_path = split1_data.get("audio_path", "")

        if not split1_audio_path:
            return None
        if not os.path.exists(split1_audio_path):
            logging.warning(f"Skip line {idx}: split1 audio not found: {split1_audio_path}")
            return None

        repeated_audio_path, repeated_duration = repeat_split1_audio(split1_audio_path, silence=SILENCE)

        metadata_out = record

        metadata_out["main"]["audio_path"] = repeated_audio_path
        metadata_out["main"]["duration"] = repeated_duration
        # metadata_out["main"]["audio_caption"] = "" #<- TODO

        # Keep split2 fully aligned with split1 content.
        metadata_out["split2"] = copy.deepcopy(split1_data)
        return metadata_out
    except Exception as e:
        logging.warning(f"Error processing line {idx}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step4 metadata generation: repeat split1 audio as new main audio, "
            "and enforce split2 to be identical to split1."
        )
    )
    parser.add_argument(
        "-i",
        "--input_jsonl",
        type=Path,
        required=True,
        help="Path to input metadata.step3_refine.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output_jsonl",
        type=Path,
        required=True,
        help="Path to output metadata.step4_repeat_main.jsonl",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already written in output_jsonl using split1.audio_path as key",
    )
    parser.add_argument("--nj", type=int, default=1, help="Number of parallel workers")
    parser.add_argument(
        "--parallel_backend",
        type=str,
        default="loky",
        choices=["threading", "loky"],
        help="joblib backend",
    )
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    if args.config_path:
        args, _ = apply_step_config(args, "step4_repeat_gen")

    global SILENCE
    SILENCE = args.silence if hasattr(args, "silence") else 0.

    runner = JsonlParallelRunner(
        input_jsonl=str(args.input_jsonl),
        output_jsonl=str(args.output_jsonl),
        process_fn=process_one,
        n_jobs=args.nj,
        backend=args.parallel_backend,
        desc=f"Step4.1 Repeat split1 with middle silence={SILENCE}",
        resume=args.resume,
        resume_key_fn=lambda rec: rec["split1"]["audio_path"],
    )
    runner.run()


if __name__ == "__main__":
    main()
