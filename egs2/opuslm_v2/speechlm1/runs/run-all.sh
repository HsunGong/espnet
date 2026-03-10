#!/bin/bash
set -euo pipefail

model="base"

. utils/parse_options.sh

train_sets=""
train_sets+="dialogue:audio_edit-prompts_v1-add_v1-input_caption-0k-30k "

valid_sets=""
valid_sets+="dialogue:audio_edit-prompts_v1-add_v1-input_caption-dev "

function create_test_sets {
    base=$1
    test_sets=""
    # test_sets+="dialogue:eval-test_clean-v2-fixed-style_emotion-${base} "
    # test_sets+="dialogue:eval-test_clean-v2-fixed-style_whisper-${base} "
    test_sets+="dialogue:eval-test_clean_audioset-v4-music_creative_edit-${base} "
    test_sets+="dialogue:eval-test_clean_audioset-v4-sing_creative_edit-${base} "
    test_sets+="dialogue:eval-test_clean_audioset-v4-sound_creative_edit-${base} "
    test_sets+="dialogue:eval-test_clean_audioset-v4-speech_creative_edit-${base} "
    export test_sets
}

# base-model
if [[ $model == "base" ]]; then
create_test_sets tgt2audio
bash launch_opuslm_stage3_sft.sh \
    --train_registered_specifier "$train_sets" \
    --valid_registered_specifier "$valid_sets" \
    --test_registered_specifier "$test_sets" \
    --train_config conf/stage2_qwen3.yaml --inference_nj 8 --inference_workers 2 \
    --resume_path /mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_350000 \
    --inference_config conf/inference_audio.yaml \
    --exp_dir exp/opuslm_v2_stage2_pretrain_base \
    --stage 3 --stop_stage 3
create_test_sets t2a_t2a
bash launch_opuslm_stage3_sft.sh \
    --train_registered_specifier "$train_sets" \
    --valid_registered_specifier "$valid_sets" \
    --test_registered_specifier "$test_sets" \
    --train_config conf/stage2_qwen3.yaml --inference_nj 8 --inference_workers 2 \
    --resume_path /mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_350000 \
    --inference_config conf/inference_audio.yaml \
    --exp_dir exp/opuslm_v2_stage2_pretrain_base \
    --stage 3 --stop_stage 3
fi

# c2a
for step in 356000 359000; do
    create_test_sets cat2split1
if [[ $model == "c2a" ]]; then
    bash launch_opuslm_stage3_sft.sh \
        --train_registered_specifier "$train_sets" \
        --valid_registered_specifier "$valid_sets" \
        --test_registered_specifier "$test_sets" \
        --inference_nj 8 --inference_workers 2 \
        --resume_path /mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_350000 \
        --train_config conf/train_stage3_ct-v2.yaml \
        --inference_config conf/inference_audio_continue.yaml \
        --inference_step $step \
        --stage 3 --stop_stage 3 \
        --exp_dir exp/ct-c2a_v2-1000k
fi
    # t2a
if [[ $model == "t2a" ]]; then
    create_test_sets t2a_t2a
    bash launch_opuslm_stage3_sft.sh \
        --train_registered_specifier "$train_sets" \
        --valid_registered_specifier "$valid_sets" \
        --test_registered_specifier "$test_sets" \
        --inference_nj 8 --inference_workers 2 \
        --resume_path /mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_350000 \
        --train_config conf/train_stage3_mt-v2.yaml \
        --inference_config conf/inference_audio.yaml \
        --inference_step $step \
        --stage 3 --stop_stage 3 \
        --exp_dir exp/ct-mt-t2a_v2-1000k
fi
done