
bash launch_opuslm_stage3_sft.sh \
    --train_registered_specifier "dialogue:vctk_vc_train_realistic dialogue:vctk_vc_train_imaginary dialogue:vctk_vc_dev_realistic dialogue:vctk_vc_dev_imaginary" \
    --valid_registered_specifier "dialogue:vctk_vc_eval1_realistic" \
    --test_registered_specifier "dialogue:vctk_vc_eval1_imaginary" \
    --exp_dir exp/opuslm_v2_stage3_sft_vctk_vc-5epoch \
    --train_config conf/train_stage3_qwen3_vc.yaml
