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

python local_eval/speech/infer_parallel.py \
  --gpus 0,1,2,3,4,5,6,7 --max-workers-per-gpu 3 --infer-script local_eval/speech/infer_cosyvoice3.py \
  --output-dir exp/cv3/test_clean/speech_edit --jsonl data/test_clean/speech_edit/*.jsonl

python local_eval/speech/infer_cv3_light.py --output-dir exp/cv3/test_clean/speech_edit --jsonl-files data/test_clean/speech_edit/{audio_effect_dereverb,style_whisper}.jsonl

python local_eval/speech/infer_parallel.py \
  --gpus 0,1,2,3,4,5,6,7 --max-workers-per-gpu 4 --infer-script local_eval/speech/infer_cosyvoice3.py \
  --output-dir data/part4/speech_edit-v2/split6/cv3 --jsonl data/part4/speech_edit-v2/split6/*.jsonl

# CUDA_VISIBLE_DEVICES=0 python -m local_eval.eval --config local_eval/eval/eval.yaml --metadata data/test_clean/speech_edit --data-dir exp/minguniaudioedit/test_clean/speech_edit --resume | tee exp/minguniaudioedit/test_clean/speech_edit.summary

python local_eval/eval_parallel.py --gpus 0,1,2,3,4,5,6,7 --max-workers-per-gpu 4 --config local_eval/eval/eval.yaml --metadata data/test_clean/speech_edit --data-dirs exp/ct-100k-default-mt/inference/*/eval-test_clean-v1-* exp/ct-100k-default-c2a/inference/*/eval-test_clean-v1-*

### special part for bagpiper
for i in data/test_clean/speech_edit/*.jsonl; do
    python local_eval/speech/assemble_dialogue.py \
    -i "$i" \
    --yaml-path data/test_clean/speech_edit/dialogues/data.yaml \
    --name-prefix eval-test_clean-v1 \
    -o data/test_clean/speech_edit/dialogues \
    --mode cat2split1 t2a_t2a a2t_t2a
done

for i in data/test_clean/speech_edit-short/*.jsonl; do
    python local_eval/speech/assemble_dialogue.py \
    -i "$i" \
    --yaml-path data/test_clean/speech_edit-short/dialogues/data.yaml \
    --name-prefix eval-test_clean-short-v2 \
    -o data/test_clean/speech_edit-short/dialogues \
    --mode cat2split1 t2a_t2a a2t_t2a
done

yq 'keys[]' data/test_clean/speech_edit/dialogues/data.yaml

python3 local_eval/speech/convert_results.py --name-prefix eval-test_clean-v1 \
  --exp-inference-dir exp/ct-100k-default-mt/inference/inference_audio_step_380000
### endregion

CUDA_VISIBLE_DEVICES=1 python -m local_eval.eval --config local_eval/eval/eval.yaml --metadata data/test_clean/speech_edit --data-dir exp/stepaudiox/test_clean/speech_edit

###### todo train

python local_eval/speech/mixup_data.py \
  --scp_dir  data/part4/speech_edit-v2/split6/cv3 \
  --jsonl_dir data/part4/speech_edit-v2 \
  --output_dir data/part4/speech_edit-v2/with_audio \
  --captioner_url http://cnode1-002:8000/v1,http://cnode1-002:8001/v1,http://cnode1-002:8002/v1,http://cnode1-002:8003/v1,http://cnode1-002:8004/v1,http://cnode1-002:8005/v1,http://cnode1-002:8006/v1,http://cnode1-002:8007/v1 \
  --nj 1024

rm data/part4/speech_edit-v2/with_audio/dialogues/data.yaml
for i in data/part4/speech_edit-v2/with_audio/*.jsonl; do
    python local_eval/speech/assemble_dialogue.py \
    -i "$i" \
    --yaml-path data/part4/speech_edit-v2/with_audio/dialogues/data.yaml \
    --name-prefix part2-speech_edit-v2 \
    -o data/part4/speech_edit-v2/with_audio/dialogues \
    --mode cat2split1 t2a_t2a a2t_t2a cat2main
done

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

python local_eval/non_speech/step3_gen.py -c local_eval/non_speech/gen_v3.yaml --style_bank_size 16 --mode speech,music,sing,sound -k 1000 --nj 512 -o data/test_clean/freeform-edit --speech data/test_clean/metadata.jsonl

rm data/test_clean/freeform-edit/dialogues/data.yaml
for i in data/test_clean/freeform-edit/*.jsonl; do
  python local_eval/non_speech/assemble_dialogue.py \
    -i "$i" \
    --yaml-path data/test_clean/freeform-edit/dialogues/data.yaml \
    --name-prefix eval-test_clean_audioset-v3 \
    -o data/test_clean/freeform-edit/dialogues \
    --mode cat2split1 t2a_t2a a2t_t2a
done

yq 'keys[]' data/test_clean/freeform-edit/dialogues/data.yaml


#### train

rm data/train_clean/audio_edit-v2/dialogues/data.yaml
for i in data/train_clean/audio_edit-v2/*.jsonl; do
    python local_eval/non_speech/assemble_dialogue.py \
        -i "$i" \
        --yaml-path data/train_clean/audio_edit-v2/dialogues/data.yaml \
        --name-prefix train_clean_audioset-v2 \
        -o data/train_clean/audio_edit-v2/dialogues \
        --mode cat2split1 cat2main t2a_t2a a2t_t2a
done

# endregion
