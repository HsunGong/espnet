#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Prepare autoencoder evaluation input from inference results."""

import argparse
import json
import logging
from pathlib import Path


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare autoencoder input from inference decode results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--decode_dir",
        type=str,
        required=True,
        help="Path to decode output directory containing results.json files",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["audio_to_text", "text_to_audio", "auto"],
        default="auto",
        help="Task type: audio_to_text, text_to_audio, or auto (auto-detect)",
    )
    parser.add_argument(
        "--original_name",
        type=str,
        default=None,
        help="Original dataset name (extracted from decode_dir if not provided)",
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
        help="Logging level",
    )
    return parser


def detect_task_type(messages):
    """Detect task type from messages by checking the last assistant message modality.

    Args:
        messages: List of [role, modality, content] tuples

    Returns:
        "audio_to_text" if last assistant message is text
        "text_to_audio" if last assistant message is audio
        None if unable to detect
    """
    # Find the last assistant message
    for role, modality, content in reversed(messages):
        if role == "assistant":
            if modality == "text":
                return "audio_to_text"
            elif modality == "audio":
                return "text_to_audio"
    return None


def extract_content(messages, task_type):
    """Extract content from the last assistant message based on task type.

    Args:
        messages: List of [role, modality, content] tuples
        task_type: "audio_to_text" or "text_to_audio"

    Returns:
        Tuple of (modality, content) or (None, None) if not found
    """
    target_modality = "text" if task_type == "audio_to_text" else "audio"

    # Find the last assistant message with target modality
    for role, modality, content in reversed(messages):
        if role == "assistant" and modality == target_modality:
            return modality, content

    return None, None


def main():
    """Main function."""
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    decode_dir = Path(args.decode_dir)
    if not decode_dir.exists():
        raise ValueError(f"Decode directory does not exist: {decode_dir}")

    # Find all results.json files
    results_files = list(decode_dir.rglob("results.json"))
    if not results_files:
        raise ValueError(f"No results.json files found in {decode_dir}")

    logging.info(f"Found {len(results_files)} results.json files")

    # Load all results
    all_results = {}
    for results_file in results_files:
        logging.info(f"Loading {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_results.update(data)

    logging.info(f"Loaded {len(all_results)} examples total")

    if not all_results:
        raise ValueError("No examples found in results.json files")

    # Auto-detect task type if needed
    task_type = args.task_type
    if task_type == "auto":
        # Use first example to detect
        first_example_id = next(iter(all_results))
        first_messages = all_results[first_example_id]
        task_type = detect_task_type(first_messages)

        if task_type is None:
            raise ValueError(
                "Could not auto-detect task type. "
                "Please specify --task_type explicitly."
            )
        logging.info(f"Auto-detected task type: {task_type}")

    # Create output directory
    output_dir = decode_dir / "autoencoder_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "input.jsonl"

    # Process examples and write JSONL
    num_written = 0
    num_skipped = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for example_id, messages in all_results.items():
            modality, content = extract_content(messages, task_type)

            if modality is None:
                logging.warning(
                    f"Skipping {example_id}: no matching content found"
                )
                num_skipped += 1
                continue

            # Create dialogue entry with single message
            # Using "user" role as input for autoencoder evaluation
            dialogue_entry = {
                "example_id": example_id,
                "messages": [["user", modality, content]]
            }

            f.write(json.dumps(dialogue_entry, ensure_ascii=False) + "\n")
            num_written += 1

    logging.info(f"Written {num_written} examples to {output_file}")
    if num_skipped > 0:
        logging.warning(f"Skipped {num_skipped} examples")

    # Determine original name
    original_name = args.original_name
    if original_name is None:
        # Extract from decode_dir: e.g., "audio_to_text_librispeech_test_clean" -> "librispeech_test_clean"
        # The decode_dir name format is: {task}_{dataset_name}
        # where task is "audio_to_text" or "text_to_audio"
        dir_name = decode_dir.name
        if dir_name.startswith("audio_to_text_"):
            original_name = dir_name[len("audio_to_text_"):]
        elif dir_name.startswith("text_to_audio_"):
            original_name = dir_name[len("text_to_audio_"):]
        else:
            # Fallback: use the whole dir name
            original_name = dir_name
        logging.info(f"Extracted original name from decode_dir: {original_name}")

    # Compute inverse task
    inverse_task = "text_to_audio" if task_type == "audio_to_text" else "audio_to_text"

    # Output the specifier: ${inverse_task}:${name}_autoencoder:${path}
    dataset_json = output_dir / "dataset.json"
    specifier = f"{inverse_task}:{original_name}_autoencoder:{dataset_json}"

    # Print specifier for shell script to capture
    print(specifier)


if __name__ == "__main__":
    main()
