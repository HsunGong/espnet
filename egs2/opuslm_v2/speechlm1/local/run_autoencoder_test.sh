inference_step=120000

for x in librispeech_test_clean librispeech_test_other mmau_test_mini_music mmau_test_mini_sound mmau_test_mini_speech; do
     echo "Running autoencoder test for ${x}"

     decode_dir="exp/opuslm_v2_stage2_pretrain_base/inference/inference_step_${inference_step}/audio_to_text_${x}/"
     bash local/test_autoencoder.sh --decode_dir "${decode_dir}"

     decode_dir="exp/opuslm_v2_stage2_pretrain_base/inference/inference_step_${inference_step}/text_to_audio_${x}/"
     bash local/test_autoencoder.sh --decode_dir "${decode_dir}"
 done