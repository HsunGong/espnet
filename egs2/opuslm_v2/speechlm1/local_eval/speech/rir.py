import numpy as np
import scipy.signal as ss
import rir_generator as rir

# size=3m -> T60 ~ 0.24s (small)
# size=8m -> T60 ~ 0.64s (medium)
# size=15m -> T60 ~ 1.2s (large)

default_rooms = {
    "small": {"L": [3.0, 3.0, 3.0], "t60": 0.24},
    "medium": {"L": [8.0, 8.0, 8.0], "t60": 0.64},
    "large": {"L": [15.0, 15.0, 15.0], "t60": 1.20},
}


def _default_positions(room_dims):
    x, y, z = room_dims
    # Keep positions away from walls and floor/ceiling.
    s = [x * 0.3, y * 0.4, max(1.2, z * 0.3)]
    r = [x * 0.7, y * 0.6, max(1.2, z * 0.25)]
    return r, s


def _to_samples_channels(audio):
    if audio.ndim == 1:
        return audio[:, None], "mono"
    if audio.ndim != 2:
        raise ValueError("audio must be 1D or 2D array")

    # Heuristic: small first dim implies channels-first (C, N).
    if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
        return audio.T, "channels_first"
    return audio, "samples_first"


def _restore_layout(audio, layout):
    if layout == "mono":
        return audio[:, 0]
    if layout == "channels_first":
        return audio.T
    return audio


def apply_rir(audio, sr: int, room_size: str):
    if audio is None:
        return None

    cfg = default_rooms.get(room_size, default_rooms["medium"])
    L = cfg["L"]
    t60 = cfg["t60"]
    nsample = max(int(sr * t60), 256)
    r, s = _default_positions(L)

    h = rir.generate(
        c=340,
        fs=sr,
        r=[r],
        s=s,
        L=L,
        reverberation_time=t60,
        nsample=nsample,
    )
    h = np.asarray(h)
    if h.ndim == 2 and h.shape[1] == 1:
        h = h[:, 0]

    audio_sc, layout = _to_samples_channels(np.asarray(audio))
    out = []
    for ch in range(audio_sc.shape[1]):
        out.append(ss.fftconvolve(audio_sc[:, ch], h, mode="full"))
    out = np.stack(out, axis=1)
    return _restore_layout(out, layout)

if __name__ == "__main__":
    src = "/mnt/home/haoranw4-andr-49167f/data/sft_data/part3_known_high_quality/audio/owsm_finetune/commonvoice_common_voice_en_19689354.mp3_000000000_000005478_eng_asr.flac"
    import soundfile as sf

    y, sr = sf.read(src)

    y_rir = apply_rir(y, sr, "medium")
    sf.write("rir_test.wav", y_rir, sr)
