#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

# region: speech
# python local_eval/speech/step0_prepare.py # only once

# if step1 changed, all need to rerun
python local_eval/speech/step1_gen.py -i data/test_clean/metadata.jsonl -o data/test_clean/speech_edit -c ./local_eval/speech/gen.yaml -k 200 --nj 256

CUDA_VISIBLE_DEVICES=0 python local_eval/speech/infer_stepaudiox.py --jsonl-files data/test_clean/speech_edit/audio_effect_dereverb.jsonl --output-dir exp/stepaudiox/test_clean/speech_edit
CUDA_VISIBLE_DEVICES=0 python local_eval/speech/infer_stepaudiox.py --jsonl-files data/test_clean/speech_edit/style_emotion.jsonl --output-dir exp/stepaudiox/test_clean/speech_edit
CUDA_VISIBLE_DEVICES=0 python local_eval/speech/infer_stepaudiox.py --jsonl-files data/test_clean/speech_edit/style_whisper.jsonl --output-dir exp/stepaudiox/test_clean/speech_edit
CUDA_VISIBLE_DEVICES=0 python local_eval/speech/infer_stepaudiox.py --jsonl-files data/test_clean/speech_edit/transcription_add_paralinguistic.jsonl --output-dir exp/stepaudiox/test_clean/speech_edit

CUDA_VISIBLE_DEVICES=2 python local_eval/speech/infer_stepaudiox.py --jsonl-files data/test_clean/speech_edit/transcription_del.jsonl --output-dir exp/stepaudiox/test_clean/speech_edit
CUDA_VISIBLE_DEVICES=2 python local_eval/speech/infer_stepaudiox.py --jsonl-files data/test_clean/speech_edit/transcription_ins.jsonl --output-dir exp/stepaudiox/test_clean/speech_edit
CUDA_VISIBLE_DEVICES=2 python local_eval/speech/infer_stepaudiox.py --jsonl-files data/test_clean/speech_edit/transcription_replace_sentence.jsonl --output-dir exp/stepaudiox/test_clean/speech_edit
CUDA_VISIBLE_DEVICES=2 python local_eval/speech/infer_stepaudiox.py --jsonl-files data/test_clean/speech_edit/transcription_sub.jsonl --output-dir exp/stepaudiox/test_clean/speech_edit

CUDA_VISIBLE_DEVICES=5 python local_eval/speech/infer_minguniaudioedit.py \
    --jsonl-files data/test_clean/speech_edit/{audio_effect_dereverb,audio_effect_pitch,audio_effect_reverb,audio_effect_speed}.jsonl \
    --output-dir exp/stepaudiox/test_clean/speech_edit
CUDA_VISIBLE_DEVICES=6 python local_eval/speech/infer_minguniaudioedit.py \
    --jsonl-files data/test_clean/speech_edit/{audio_effect_volume,style_emotion,style_whisper,transcription_add_paralinguistic}.jsonl \
    --output-dir exp/stepaudiox/test_clean/speech_edit
CUDA_VISIBLE_DEVICES=7 python local_eval/speech/infer_minguniaudioedit.py \
    --jsonl-files data/test_clean/speech_edit/{transcription_del,transcription_ins,transcription_replace_sentence,transcription_sub}.jsonl \
    --output-dir exp/stepaudiox/test_clean/speech_edit

for i in data/test_clean/speech_edit/*.jsonl; do
    python local_eval/speech/assemble_dialogue.py \
    -i "$i" \
    --yaml-path data/test_clean/speech_edit/dialogues/data.yaml \
    --name-prefix eval-test_clean-v1 \
    -o data/test_clean/speech_edit/dialogues \
    --mode cat2split1 t2a_t2a a2t_t2a
done

yq 'keys[]' data/test_clean/speech_edit/dialogues/data.yaml

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
