#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

# region: speech
# python local_eval/speech/step0_prepare.py # only once

# if step1 changed, all need to rerun
python local_eval/speech/step1_gen.py -i data/test_clean/metadata.jsonl -o data/test_clean/speech_edit -c ./local_eval/speech/gen.yaml -k 200 --nj 256

# gpu-util=0.3 -> support 3 parallel jobs
python local_eval/speech/infer_parallel.py \
  --gpus 0,1,2 --max-workers-per-gpu 3 --infer-script local_eval/speech/infer_stepaudiox.py \
  --output-dir exp/stepaudiox/test_clean/speech_edit --jsonl data/test_clean/speech_edit/*.jsonl
python local_eval/speech/infer_parallel.py \
  --gpus 0,1,2 --max-workers-per-gpu 3 --infer-script local_eval/speech/infer_stepaudiox.py \
  --output-dir exp/stepaudiox/test_clean/speech_edit-short --jsonl data/test_clean/speech_edit-short/*.jsonl

python local_eval/speech/infer_parallel.py \
  --gpus 0,1,2,3,4,5,6,7 --max-workers-per-gpu 1 --infer-script local_eval/speech/infer_minguniaudioedit.py \
  --output-dir exp/minguniaudioedit/test_clean/speech_edit-short --jsonl data/test_clean/speech_edit-short/*.jsonl


CUDA_VISIBLE_DEVICES=7 python local_eval/speech/infer_cosyvoice3.py --output-dir exp/cv3/test_clean/speech_edit --jsonl-files data/test_clean/speech_edit/{audio_effect_dereverb,style_whisper}.jsonl
CUDA_VISIBLE_DEVICES=6 python local_eval/speech/infer_cosyvoice3.py --output-dir exp/cv3/test_clean/speech_edit --jsonl-files data/test_clean/speech_edit/{audio_effect_pitch,transcription_add_paralinguistic}.jsonl
CUDA_VISIBLE_DEVICES=5 python local_eval/speech/infer_cosyvoice3.py --output-dir exp/cv3/test_clean/speech_edit --jsonl-files data/test_clean/speech_edit/{audio_effect_reverb,transcription_del}.jsonl
CUDA_VISIBLE_DEVICES=4 python local_eval/speech/infer_cosyvoice3.py --output-dir exp/cv3/test_clean/speech_edit --jsonl-files data/test_clean/speech_edit/{audio_effect_speed,transcription_ins}.jsonl
CUDA_VISIBLE_DEVICES=3 python local_eval/speech/infer_cosyvoice3.py --output-dir exp/cv3/test_clean/speech_edit --jsonl-files data/test_clean/speech_edit/{audio_effect_volume,transcription_replace_sentence}.jsonl
CUDA_VISIBLE_DEVICES=2 python local_eval/speech/infer_cosyvoice3.py --output-dir exp/cv3/test_clean/speech_edit --jsonl-files data/test_clean/speech_edit/{style_emotion,transcription_sub}.jsonl


### special part for bagpiper
for i in data/test_clean/speech_edit/*.jsonl; do
    python local_eval/speech/assemble_dialogue.py \
    -i "$i" \
    --yaml-path data/test_clean/speech_edit/dialogues/data.yaml \
    --name-prefix eval-test_clean-v1 \
    -o data/test_clean/speech_edit/dialogues \
    --mode cat2split1 t2a_t2a a2t_t2a
done

yq 'keys[]' data/test_clean/speech_edit/dialogues/data.yaml

python3 local_eval/speech/convert_results.py --name-prefix eval-test_clean-v1 \
  --exp-inference-dir exp/ct-100k-default-mt/inference/inference_audio_step_380000
### endregion

CUDA_VISIBLE_DEVICES=1 python -m local_eval.eval --config local_eval/eval/eval.yaml --metadata data/test_clean/speech_edit --data-dir exp/stepaudiox/test_clean/speech_edit

# endregion

# region: non-speech

python local_eval/non_speech/step1_gen.py --speech data/test_clean/metadata.jsonl --sound data/mmau/metadata.sound.jsonl --music data/mmau/metadata.music.jsonl -o data/test_clean/audio_edit -c ./local_eval/non_speech/gen.yaml -k 200 --nj 256

for i in data/test_clean/audio_edit/*.jsonl; do
    python local_eval/non_speech/assemble_dialogue.py \
        -i "$i" \
        --yaml-path data/test_clean/audio_edit/dialogues/data.yaml \
        --name-prefix eval-test_clean_mmau-v1 \
        -o data/test_clean/audio_edit/dialogues \
        --mode cat2split1 t2a_t2a a2t_t2a
done

yq 'keys[]' data/test_clean/audio_edit/dialogues/data.yaml

# endregion
