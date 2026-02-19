from .audio_loader import LhotseAudioReader, ArkiveAudioReader, KaldiAudioReader
from .text_loader import TextReader, NumericReader, ArkiveTextReader
from .dialogue_loader import DialogueReader, ArkiveDialogueLoader

ALL_DATA_LOADERS = {
    "lhotse_audio": LhotseAudioReader,
    "arkive_audio": ArkiveAudioReader,
    "text": TextReader,
    "arkive_text": ArkiveTextReader,
    "dialogue": DialogueReader,
    "arkive_dialogue": ArkiveDialogueLoader,
    "kaldi_audio": KaldiAudioReader,
    "numeric": NumericReader,
}

__all__ = [
    "ALL_DATA_LOADERS",
]
