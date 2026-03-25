#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""SpeechLM inference entrypoint for assistant-prefix continuation decoding."""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
import yaml

from espnet2.speechlm.dataloader.iterator import DataIteratorFactory
from espnet2.speechlm.model import _all_job_types
from espnet2.speechlm.model.speechlm.lm.parallel import (
    ParallelHFModelDecodeWithPrefix,
)
from espnet2.speechlm.model.speechlm.speechlm_job import (
    SpeechLMPreprocessorDecodeWithPrefix,
)
from espnet2.speechlm.utils.data import to_device


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="SpeechLM Prefix-Continuation Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        required=True,
        help="Path to inference configuration file",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint to load",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/inference_mp"),
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--test-unregistered-specifier",
        type=str,
        default=None,
        help="Unregistered test data specifier " "(e.g., 'asr:librispeech:test.json')",
    )
    parser.add_argument(
        "--test-registered-specifier",
        type=str,
        default=None,
        help="Registered test data specifier " "(e.g., 'asr:librispeech')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for inference",
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="GPU rank in the whole inference job",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        help="number of GPUs in the whole inference job",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible inference",
    )
    parser.add_argument(
        "--add-generation-prompt",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=None,
        help="Fallback to regular generation mode instead of continuation mode",
    )
    parser.add_argument(
        "--continuation-prefix-ratio",
        type=float,
        default=None,
        help="Fraction of the last assistant audio frames to keep as prefix",
    )
    return parser


def setup_worker_logger(rank: int) -> logging.Logger:
    """Set up logger for worker process."""
    logger = logging.getLogger(f"inference_with_prefix_worker_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        f"[Worker-{rank}] [%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["module"]
    model.load_state_dict(state_dict, strict=True)
    return model


def resolve_inference_settings(args, inference_config):
    """Resolve CLI/config continuation settings."""
    if args.add_generation_prompt is not None:
        add_generation_prompt = args.add_generation_prompt
    else:
        add_generation_prompt = inference_config.get("add_generation_prompt", False)

    if args.continuation_prefix_ratio is not None:
        continuation_prefix_ratio = args.continuation_prefix_ratio
    else:
        continuation_prefix_ratio = inference_config.get(
            "continuation_prefix_ratio", 0.5
        )

    inference_config["add_generation_prompt"] = add_generation_prompt
    inference_config["continuation_prefix_ratio"] = continuation_prefix_ratio
    return add_generation_prompt, continuation_prefix_ratio


def build_prefix_model_and_preprocessor(train_config, inference_config):
    """Instantiate the prefix-aware model and preprocessor."""
    job_template_class = _all_job_types[train_config["job_type"]]
    job_template = job_template_class(train_config, is_train=False)

    add_generation_prompt = inference_config["add_generation_prompt"]
    continuation_prefix_ratio = inference_config["continuation_prefix_ratio"]

    model_config = train_config["model"]
    model = ParallelHFModelDecodeWithPrefix(
        model_hf_tag=model_config["model_hf_tag"],
        multimodal_io=job_template.multimodal_io,
        vocab=job_template.vocab,
        vocab_intervals=job_template.vocab_intervals,
        **model_config["model_conf"],
    )

    processor_config = train_config["preprocessor"]
    multimodal_io = {
        io_name: io.copy_for_worker() for io_name, io in job_template.multimodal_io.items()
    }
    preprocessor = SpeechLMPreprocessorDecodeWithPrefix(
        is_train=False,
        multimodal_io=multimodal_io,
        vocab=job_template.vocab,
        vocab_intervals=job_template.vocab_intervals,
        audio_input=processor_config["audio_input"],
        audio_output=processor_config["audio_output"],
        loss_region=processor_config["loss_region"],
        batchfy_method=train_config["data_loading"].get("batchfy_method", "bucket"),
        audio_cfg=processor_config.get("audio_cfg", 0.0),
        batch_length=train_config["data_loading"].get("batch_size", -1),
        add_generation_prompt=add_generation_prompt,
        continuation_prefix_ratio=continuation_prefix_ratio,
    )
    return model, preprocessor


@torch.no_grad()
def inference_worker(
    rank: int,
    world_size: int,
    train_config_path: Path,
    inference_config_path: Path,
    model_checkpoint_path: Path,
    unregistered_specifier: str,
    registered_specifier: str,
    output_dir: Path,
    seed: int,
    add_generation_prompt_override,
    continuation_prefix_ratio_override,
):
    """Worker process for prefix-aware inference with data sharding."""
    logger = setup_worker_logger(rank)
    logger.info(f"Starting inference worker (rank {rank}/{world_size})")

    torch.cuda.set_device("cuda:0")

    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)

    with open(inference_config_path, "r") as f:
        inference_config = yaml.safe_load(f)

    class ArgsProxy:
        add_generation_prompt = add_generation_prompt_override
        continuation_prefix_ratio = continuation_prefix_ratio_override

    _, _ = resolve_inference_settings(ArgsProxy, inference_config)

    model, preprocessor = build_prefix_model_and_preprocessor(
        train_config, inference_config
    )

    model = load_checkpoint(model, model_checkpoint_path)
    model.prepare_inference()
    dtype = inference_config.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype)
    model = model.to(device="cuda", dtype=dtype).eval()

    iterator_factory = DataIteratorFactory(
        unregistered_specifier=unregistered_specifier,
        registered_specifier=registered_specifier,
        collate_fn=preprocessor.collate_fn,
        num_workers=0,
        rank=rank,
        world_size=world_size,
        sequential_load=True,
    )

    output_dir = output_dir / f"inference_rank{rank}"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "results.json"

    test_iterator = iterator_factory.build_iter()
    results = dict()
    logger.info("Starting prefix-continuation inference on data shard")

    for idx, sample in enumerate(test_iterator):
        sample = to_device(sample, "cuda", dtype=dtype)
        task, data_name, example_id = sample.pop("keys")[0]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        logger.info(f"Processing sample {idx}: {task}/{data_name}/{example_id}")
        messages, _ = model.inference(inference_config, **sample)

        output_messages = []
        for seg_idx, (role, modality, content) in enumerate(messages):
            if role == "prefix":
                logger.info(f"Skip prefix payload for {example_id} segment {seg_idx}")
                continue

            if modality == "audio":
                audio, length, sample_rate = content
                audio, length = audio[0], length[0]
                audio = audio.cpu().float().numpy()

                content = output_dir / f"{example_id}_segment{len(output_messages) + 1}.wav"
                sf.write(content, audio.T, sample_rate)
                content = str(content)

            logger.info(
                f"Segment {seg_idx}, role={role}, modality={modality}, content={content}"
            )
            output_messages.append([role, modality, content])

        results[example_id] = output_messages
        with open(output_file, "wb") as writer:
            writer.write(
                json.dumps(
                    results, indent=4, ensure_ascii=False, sort_keys=False
                ).encode("utf_8")
            )


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires GPU.")
        sys.exit(1)

    if not args.test_registered_specifier and not args.test_unregistered_specifier:
        parser.error(
            "Provide either --test-registered-specifier or "
            "--test-unregistered-specifier"
        )
    if args.test_registered_specifier and args.test_unregistered_specifier:
        parser.error(
            "Provide only one of --test-registered-specifier or "
            "--test-unregistered-specifier"
        )

    specifier = args.test_registered_specifier or args.test_unregistered_specifier
    output_dir = args.output_dir / specifier.replace(":", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    mp.set_start_method("spawn", force=True)

    processes = []
    args.rank -= 1
    start_rank = args.rank * args.num_workers
    end_rank = (args.rank + 1) * args.num_workers
    for rank in range(start_rank, end_rank):
        p = mp.Process(
            target=inference_worker,
            args=(
                rank,
                args.world_size * args.num_workers,
                args.train_config,
                args.inference_config,
                args.model_checkpoint,
                args.test_unregistered_specifier or "",
                args.test_registered_specifier or "",
                output_dir,
                args.seed,
                args.add_generation_prompt,
                args.continuation_prefix_ratio,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(60)

    for p in processes:
        p.join()

    print("All workers completed!")


if __name__ == "__main__":
    main()
