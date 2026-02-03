#!/bin/bash

bash launch_opuslm_stage3_sft.sh \
    --train_registered_specifier "dialogue:librimix_spk1_enh-v3_train-10k_keep dialogue:librimix_spk1_enh-v3_train-10k_remove dialogue:librimix_spk1_enh-v3_train-20k_keep dialogue:librimix_spk1_enh-v3_train-20k_remove dialogue:librimix_spk1_enh-v3_train-30k_keep dialogue:librimix_spk1_enh-v3_train-30k_remove" \
    --valid_registered_specifier "dialogue:librimix_spk1_enh-v3_test100" \
    --test_registered_specifier "dialogue:librimix_spk1_enh-v3_test100" \
    --train_config conf/train_stage3_qwen3_enh-v3.yaml \
    --resume_path /mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_350000 \
    --exp_dir exp/opuslm_v2_stage3_sft_librimix_enh-v3 \
    --stop_stage 3 \
    --inference_step 351881 \
    "$@"

# --inference_step 350545 --stage 4 --exp_dir exp/opuslm_v2_stage3_sft_vctk_vc-5epoch