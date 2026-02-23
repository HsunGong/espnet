from __future__ import annotations

import os
import tempfile
import unittest
import importlib.util

from utils import load_config, render_template


class ConfigTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("yaml") is not None, "yaml is not installed")
    def test_env_expansion(self) -> None:
        os.environ["EVAL_TEST_URL"] = "http://localhost:1234/v1"
        content = "scorers:\n  demo:\n    url: ${EVAL_TEST_URL}\n"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(content)
            path = f.name
        cfg = load_config(path)
        self.assertEqual(cfg["scorers"]["demo"]["url"], "http://localhost:1234/v1")

    @unittest.skipUnless(importlib.util.find_spec("jinja2") is not None, "jinja2 is not installed")
    def test_jinja_render(self) -> None:
        text = render_template("Hello {{ name }}", {"name": "world"})
        self.assertEqual(text, "Hello world")


if __name__ == "__main__":
    unittest.main()
