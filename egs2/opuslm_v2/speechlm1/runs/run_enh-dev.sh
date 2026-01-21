
bash launch_opuslm_stage3_sft.sh \
    train_registered_specifier="dialogue:librimix_spk1_enh-v2_dev_realistic dialogue:librimix_spk1_enh-v2_dev_imaginary dialogue:librimix_spk1_simul-v2_dev_realistic dialogue:librimix_spk1_simul-v2_dev_imaginary"
    --valid_registered_specifier "dialogue:librimix_spk1_enh-v2_test100_realistic dialogue:librimix_spk1_simul-v2_test100_realistic" \
    --test_registered_specifier "dialogue:librimix_spk1_enh-v2_test100_realistic dialogue:librimix_spk1_simul-v2_test100_realistic" \
    --train_config conf/train_stage3_qwen3_enh.yaml \
    --exp_dir exp/opuslm_v2_stage3_sft_librimix_enh-v2 "$@"
