from __future__ import annotations

import unittest

from scorers.registry import SCORER_SPECS


class RegistryTests(unittest.TestCase):
    def test_new_scorers_registered(self) -> None:
        for name in (
            "llm_judge_caption_llm",
            "llm_judge_gemini",
            "llm_judge_openai",
            "pseudo_mos",
            "emotion_modelscope",
            "speaker_similarity_wespeaker",
        ):
            self.assertIn(name, SCORER_SPECS)

if __name__ == "__main__":
    unittest.main()
