from __future__ import annotations

import unittest

from utils import coerce_bool, extract_first_float, normalize_text, try_parse_json


class UtilsTests(unittest.TestCase):
    def test_extract_first_float(self) -> None:
        self.assertAlmostEqual(extract_first_float("speed_1.25"), 1.25)
        self.assertIsNone(extract_first_float("no-number"))

    def test_try_parse_json(self) -> None:
        parsed = try_parse_json('{"valid": true, "score": 0.8}')
        self.assertTrue(parsed["valid"])
        self.assertAlmostEqual(parsed["score"], 0.8)

    def test_coerce_bool(self) -> None:
        self.assertTrue(coerce_bool("true"))
        self.assertFalse(coerce_bool("no"))

    def test_normalize_text(self) -> None:
        self.assertEqual(normalize_text("Hi,   THERE!"), "hi there")


if __name__ == "__main__":
    unittest.main()
