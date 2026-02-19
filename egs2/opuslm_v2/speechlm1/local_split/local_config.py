#!/usr/bin/env python3

from argparse import Namespace

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fin:
        return yaml.safe_load(fin)


def apply_step_config(args: Namespace, step_key: str) -> tuple[Namespace, dict]:
    config = load_config(args.config_path)
    for key, value in config[step_key].items():
        setattr(args, key, value)
    return args, config
