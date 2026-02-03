import matplotlib.pyplot as plt
import numpy as np
import os
# import scipy.io.wavfile as wavfile
import scipy.signal as signal
import soundfile as sf

# audio_path = '/work/nvme/bbjs/xgong4/opuslm/prep/inference_step_260000/text_to_audio_mmau_test_mini_music/inference_rank5/75608263-e320-4823-8c62-1c650a0f37ca_segment1.wav'
# audio_path = 'visualization/speech.wav'
# audio_path = "/work/nvme/bbjs/xgong4/opuslm/prep/inference_step_260000/text_to_audio_mmau_test_mini_music/inference_rank6/13a1d562-8f37-4991-9459-d30f6c12009f_segment1.wav"

audio_path = "/mnt/home/haoranw4-andr-49167f/data/sft_data/part2_pretrain_curation/audio/stage4_filtering_sound_gen_sft/yt8m/JvlsGZ2ZsaE_140_10_0000.flac"
output_dir = './chat_output/visualization-t2a'
audio_path = "/mnt/home/haoranw4-andr-49167f/data/sft_data/part2_pretrain_curation/audio/gemini_pilot/jamendo/17_223917_chunk.wav"
output_dir = './chat_output/visualization-a2t'

os.makedirs(output_dir, exist_ok=True)

# Load audio using scipy
y, sr = sf.read(audio_path)

# Convert to float if necessary (scipy returns int types usually)


def to_float_mono(sr: int, y: np.ndarray) -> np.ndarray:
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Convert to float
    if y.dtype.kind == "i":
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    elif y.dtype.kind == "u":
        y = (y.astype(np.float32) - np.iinfo(y.dtype).max / 2) / (np.iinfo(y.dtype).max / 2)
    else:
        y = y.astype(np.float32)

    # Handle NaN/Inf
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y

y = to_float_mono(sr, y)

plt.figure(figsize=(20,1))
frequencies, times, spec = signal.spectrogram(y, sr)
spec_db = 10 * np.log10(spec + 1e-10)

plt.pcolormesh(times, frequencies / 1000.0, spec_db, shading="gouraud")
plt.gca().yaxis.set_visible(False)
plt.xticks([])
    
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectrogram.png"), dpi=400, bbox_inches="tight")
plt.close()

print(f"Spectrogram saved to: {os.path.join(output_dir, 'spectrogram.png')}")
