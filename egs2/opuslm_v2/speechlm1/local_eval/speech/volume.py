
def apply_volume(y, sr, delta):
    import pyloudnorm as pyln
    if y.ndim == 1:
        y = y[np.newaxis, :]
    
    # 5db -> 10db as maxmisze -> scale by 1/4
    delta = 10 * max(delta / 5, 1) ** (1 / 4)

    duration = y.shape[1] / sr

    # measure the loudness first
    meter = pyln.Meter(
        sr, block_size=min(0.4, duration - 1e-10)
    )  # create BS.1770 meter
    loudness = meter.integrated_loudness(y.T)

    # loudness normalize audio to target LUFS. We will ignore the warnings related to
    # clipping the audio.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loudness_normalized_audio = pyln.normalize.loudness(y.T, loudness, loudness + delta)

    return loudness_normalized_audio.T.squeeze()

import sys

if __name__ == '__main__':
    import argparse
    import os
    import numpy as np
    import soundfile as sf
    import warnings

    input_audio = sys.argv[1]
    output_audio = sys.argv[2]
    delta_db = float(sys.argv[3])

    y, sr = sf.read(input_audio)
    y_adjusted = apply_volume(y, sr, delta_db)
    sf.write(output_audio, y_adjusted, sr)
    