#!/bin/bash

train_sets=""
train_sets+="dialogue:sft-part2_4-novad-music.min_0.max_10-dialogue.step0.main2main "
train_sets+="dialogue:sft-part2_4-novad-music.min_10-dialogue.step0.main2main "
train_sets+="dialogue:sft-part2_4-novad-sound.min_0.max_7-dialogue.step0.main2main "
train_sets+="dialogue:sft-part2_4-novad-sound.min_10-dialogue.step0.main2main "
train_sets+="dialogue:sft-part2_4-novad-sound.min_7.max_10-dialogue.step0.main2main "
train_sets+="dialogue:sft-part2_4-novad-speech.min_20.max_25-dialogue.step0.main2main "
train_sets+="dialogue:sft-part2_4-novad-speech.min_3.max_5-dialogue.step0.main2main "
train_sets+="dialogue:sft-part2_4-novad-speech.min_5.max_8-dialogue.step0.main2main "
train_sets+="dialogue:sft-part2_4-novad-speech.min_8.max_20-dialogue.step0.main2main "
train_sets+="dialogue:sft-part2_4-vad-music.min_10-dialogue.step1.main2main "
train_sets+="dialogue:sft-part2_4-vad-sound.min_10-dialogue.step1.main2main "
train_sets+="dialogue:sft-part2_4-vad-sound.min_7.max_10-dialogue.step1.main2main "
train_sets+="dialogue:sft-part2_4-vad-speech.min_20.max_25-dialogue.step1.main2main "
train_sets+="dialogue:sft-part2_4-vad-speech.min_25-dialogue.step1.main2main "
train_sets+="dialogue:sft-part2_4-vad-speech.min_5.max_8-dialogue.step1.main2main "

valid_sets=""
valid_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step1.main2main "
valid_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step1.main2main "

test_sets=""
test_sets+=""
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step8.main2split1.default "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step8.main2split1.default "

# just for debugging -> check the loss
bash launch_opuslm_stage3_sft.sh \
    --train_registered_specifier "$train_sets" \
    --valid_registered_specifier "$valid_sets" \
    --test_registered_specifier "$test_sets" \
    --inference_nj 8 --inference_workers 2 \
    --resume_path /mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_350000 \
    --train_config conf/train_stage3_pretrain.yaml \
    --inference_config conf/inference_audio_continue.yaml \
    --exp_dir exp/ct-100k-default-pretrain \
    "$@"
