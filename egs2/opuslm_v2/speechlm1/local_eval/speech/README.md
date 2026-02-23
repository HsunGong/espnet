# Use librispeech test-clean to construct the test dataset

- **Available Tags**
<table>
  <tr>
    <td rowspan="8" style="vertical-align: middle; text-align:center;" align="center">emotion</td>
    <td align="center"><b>happy</b></td>
    <td align="center">Expressing happiness</td>
    <td align="center"><b>angry</b></td>
    <td align="center">Expressing anger</td>
  </tr>
  <tr>
    <td align="center"><b>sad</b></td>
    <td align="center">Expressing sadness</td>
    <td align="center"><b>fear</b></td>
    <td align="center">Expressing fear</td>
  </tr>
  <tr>
    <td align="center"><b>surprised</b></td>
    <td align="center">Expressing surprise</td>
    <td align="center"><b>confusion</b></td>
    <td align="center">Expressing confusion</td>
  </tr>
  <tr>
    <td align="center"><b>empathy</b></td>
    <td align="center">Expressing empathy and understanding</td>
    <td align="center"><b>embarrass</b></td>
    <td align="center">Expressing embarrassment</td>
  </tr>
  <tr>
    <td align="center"><b>excited</b></td>
    <td align="center">Expressing excitement and enthusiasm</td>
    <td align="center"><b>depressed</b></td>
    <td align="center">Expressing a depressed or discouraged mood</td>
  </tr>
  <tr>
    <td align="center"><b>admiration</b></td>
    <td align="center">Expressing admiration or respect</td>
    <td align="center"><b>coldness</b></td>
    <td align="center">Expressing coldness and indifference</td>
  </tr>
  <tr>
    <td align="center"><b>disgusted</b></td>
    <td align="center">Expressing disgust or aversion</td>
    <td align="center"><b>humour</b></td>
    <td align="center">Expressing humor or playfulness</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td rowspan="17" style="vertical-align: middle; text-align:center;" align="center">speaking style</td>
    <td align="center"><b>serious</b></td>
    <td align="center">Speaking in a serious or solemn manner</td>
    <td align="center"><b>arrogant</b></td>
    <td align="center">Speaking in an arrogant manner</td>
  </tr>
  <tr>
    <td align="center"><b>child</b></td>
    <td align="center">Speaking in a childlike manner</td>
    <td align="center"><b>older</b></td>
    <td align="center">Speaking in an elderly-sounding manner</td>
  </tr>
  <tr>
    <td align="center"><b>girl</b></td>
    <td align="center">Speaking in a light, youthful feminine manner</td>
    <td align="center"><b>pure</b></td>
    <td align="center">Speaking in a pure, innocent manner</td>
  </tr>
  <tr>
    <td align="center"><b>sister</b></td>
    <td align="center">Speaking in a mature, confident feminine manner</td>
    <td align="center"><b>sweet</b></td>
    <td align="center">Speaking in a sweet, lovely manner</td>
  </tr>
  <tr>
    <td align="center"><b>exaggerated</b></td>
    <td align="center">Speaking in an exaggerated, dramatic manner</td>
    <td align="center"><b>ethereal</b></td>
    <td align="center">Speaking in a soft, airy, dreamy manner</td>
  </tr>
  <tr>
    <td align="center"><b>whisper</b></td>
    <td align="center">Speaking in a whispering, very soft manner</td>
    <td align="center"><b>generous</b></td>
    <td align="center">Speaking in a hearty, outgoing, and straight-talking manner</td>
  </tr>
  <tr>
    <td align="center"><b>recite</b></td>
    <td align="center">Speaking in a clear, well-paced, poetry-reading manner</td>
    <td align="center"><b>act_coy</b></td>
    <td align="center">Speaking in a sweet, playful, and endearing manner</td>
  </tr>
  <tr>
    <td align="center"><b>warm</b></td>
    <td align="center">Speaking in a warm, friendly manner</td>
    <td align="center"><b>shy</b></td>
    <td align="center">Speaking in a shy, timid manner</td>
  </tr>
  <tr>
    <td align="center"><b>comfort</b></td>
    <td align="center">Speaking in a comforting, reassuring manner</td>
    <td align="center"><b>authority</b></td>
    <td align="center">Speaking in an authoritative, commanding manner</td>
  </tr>
  <tr>
    <td align="center"><b>chat</b></td>
    <td align="center">Speaking in a casual, conversational manner</td>
    <td align="center"><b>radio</b></td>
    <td align="center">Speaking in a radio-broadcast manner</td>
  </tr>
  <tr>
    <td align="center"><b>soulful</b></td>
    <td align="center">Speaking in a heartfelt, deeply emotional manner</td>
    <td align="center"><b>gentle</b></td>
    <td align="center">Speaking in a gentle, soft manner</td>
  </tr>
  <tr>
    <td align="center"><b>story</b></td>
    <td align="center">Speaking in a narrative, audiobook-style manner</td>
    <td align="center"><b>vivid</b></td>
    <td align="center">Speaking in a lively, expressive manner</td>
  </tr>
  <tr>
    <td align="center"><b>program</b></td>
    <td align="center">Speaking in a show-host/presenter manner</td>
    <td align="center"><b>news</b></td>
    <td align="center">Speaking in a news broadcasting manner</td>
  </tr>
  <tr>
    <td align="center"><b>advertising</b></td>
    <td align="center">Speaking in a polished, high-end commercial voiceover manner</td>
    <td align="center"><b>roar</b></td>
    <td align="center">Speaking in a loud, deep, roaring manner</td>
  </tr>
  <tr>
    <td align="center"><b>murmur</b></td>
    <td align="center">Speaking in a quiet, low manner</td>
    <td align="center"><b>shout</b></td>
    <td align="center">Speaking in a loud, sharp, shouting manner</td>
  </tr>
  <tr>
    <td align="center"><b>deeply</b></td>
    <td align="center">Speaking in a deep and low-pitched tone</td>
    <td align="center"><b>loudly</b></td>
    <td align="center">Speaking in a loud and high-pitched tone</td>
  </tr>
  <tr>
  </tr>
  <tr>
  </tr>
  <tr>
  <td rowspan="11" style="vertical-align: middle; text-align:center;" align="center">paralinguistic</td>
    <td align="center"><b>[sigh]</b></td>
    <td align="center">Sighing sound</td>
    <td align="center"><b>[inhale]</b></td>
    <td align="center">Inhaling sound</td>
  </tr>

  <tr>
    <td align="center"><b>[laugh]</b></td>
    <td align="center">Laughter sound</td>
    <td align="center"><b>[chuckle]</b></td>
    <td align="center">Chuckling sound</td>
  </tr>

  <tr>
    <td align="center"><b>[exhale]</b></td>
    <td align="center">Exhaling sound</td>
    <td align="center"><b>[clears throat]</b></td>
    <td align="center">Throat clearing sound</td>
  </tr>

  <tr>
    <td align="center"><b>[snort]</b></td>
    <td align="center">Snorting sound</td>
    <td align="center"><b>[giggle]</b></td>
    <td align="center">Giggling sound</td>
  </tr>

  <tr>
    <td align="center"><b>[cough]</b></td>
    <td align="center">Coughing sound</td>
    <td align="center"><b>[breath]</b></td>
    <td align="center">Breathing sound</td>
  </tr>

  <tr>
    <td align="center"><b>[uhm]</b></td>
    <td align="center">Hesitation sound: "Uhm"</td>
    <td align="center"><b>[Confirmation-en]</b></td>
    <td align="center">Confirming: "En"</td>
  </tr>

  <tr>
    <td align="center"><b>[Surprise-oh]</b></td>
    <td align="center">Expressing surprise: "Oh"</td>
    <td align="center"><b>[Surprise-ah]</b></td>
    <td align="center">Expressing surprise: "Ah"</td>
  </tr>

  <tr>
    <td align="center"><b>[Surprise-wa]</b></td>
    <td align="center">Expressing surprise: "Wa"</td>
    <td align="center"><b>[Surprise-yo]</b></td>
    <td align="center">Expressing surprise: "Yo"</td>
  </tr>

  <tr>
    <td align="center"><b>[Dissatisfaction-hnn]</b></td>
    <td align="center">Dissatisfied sound: "Hnn"</td>
    <td align="center"><b>[Question-ei]</b></td>
    <td align="center">Questioning: "Ei"</td>
  </tr>

  <tr>
    <td align="center"><b>[Question-ah]</b></td>
    <td align="center">Questioning: "Ah"</td>
    <td align="center"><b>[Question-en]</b></td>
    <td align="center">Questioning: "En"</td>
  </tr>

  <tr>
    <td align="center"><b>[Question-yi]</b></td>
    <td align="center">Questioning: "Yi"</td>
    <td align="center"><b>[Question-oh]</b></td>
    <td align="center">Questioning: "Oh"</td>
  </tr>
</table>
 
## Feature Requests & Wishlist
💡 We welcome all ideas for new features! If you'd like to see a feature added to the project, please start a discussion in our [Discussions](https://github.com/stepfun-ai/Step-Audio-EditX/discussions) section.

We'll be collecting community feedback here and will incorporate popular suggestions into our future development plans. Thank you for your contribution!

## Demos

<table>
  <tr>
    <th style="vertical-align : middle;text-align: center">Task</th>
    <th style="vertical-align : middle;text-align: center">Text</th>
    <th style="vertical-align : middle;text-align: center">Source</th>
    <th style="vertical-align : middle;text-align: center">Edited</th>
  </tr>

  <tr>
    <td align="center"> Emotion-Fear</td>
    <td align="center"> 我总觉得，有人在跟着我，我能听到奇怪的脚步声。</td>
    <td align="center">

  [fear_zh_female_prompt.webm](https://github.com/user-attachments/assets/a088c059-032c-423f-81d6-3816ba347ff5) 
  </td>
    <td align="center">
      
  [fear_zh_female_output.webm](https://github.com/user-attachments/assets/917494ac-5913-4949-8022-46cf55ca05dd)
  </td>
  </tr>


  <tr>
    <td align="center"> Style-Whisper</td>
    <td align="center"> 比如在工作间隙，做一些简单的伸展运动，放松一下身体，这样，会让你更有精力。</td>
    <td align="center">
      
  [whisper_prompt.webm](https://github.com/user-attachments/assets/ed9e22f1-1bac-417b-913a-5f1db31f35c9)
  </td>
    <td align="center">
      
  [whisper_output.webm](https://github.com/user-attachments/assets/e0501050-40db-4d45-b380-8bcc309f0b5f)
  </td>
  </tr>

  <tr>
    <td align="center"> Style-Act_coy</td>
    <td align="center"> 我今天想喝奶茶，可是不知道喝什么口味，你帮我选一下嘛，你选的都好喝～</td>
    <td align="center">

  [act_coy_prompt.webm](https://github.com/user-attachments/assets/74d60625-5b3c-4f45-becb-0d3fe7cc4b3f)
  </td>
    <td align="center"> 

  [act_coy_output.webm](https://github.com/user-attachments/assets/b2f74577-56c2-4997-afd6-6bf47d15ea51)
  </td>
  </tr>


  <tr>
    <td align="center"> Paralinguistics</td>
    <td align="center"> 你这次又忘记带钥匙了 [Dissatisfaction-hnn]，真是拿你没办法。</td>
    <td align="center">
      
  [paralingustic_prompt.webm](https://github.com/user-attachments/assets/21e831a3-8110-4c64-a157-60e0cf6735f0)
  </td>
    <td align="center">
      
  [paralingustic_output.webm](https://github.com/user-attachments/assets/a82f5a40-c6a3-409b-bbe6-271180b20d7b)
  </td>
  </tr>


  <tr>
    <td align="center"> Denoising</td>
    <td align="center"> Such legislation was clarified and extended from time to time thereafter. No, the man was not drunk, he wondered how we got tied up with this stranger. Suddenly, my reflexes had gone. It's healthier to cook without sugar.</td>
    <td align="center">
      
  [denoising_prompt.webm](https://github.com/user-attachments/assets/70464bf4-ebde-44a3-b2a6-8c292333319b)
  </td>
    <td align="center">
      
  [denoising_output.webm](https://github.com/user-attachments/assets/7cd0ae8d-1bf0-40fc-9bcd-f419bd4b2d21)
  </td>
  </tr>

  <tr>
    <td align="center"> Speed-Faster</td>
    <td align="center"> 上次你说鞋子有点磨脚，我给你买了一双软软的鞋垫。</td>
    <td align="center">
      
  [speed_faster_prompt.webm](https://github.com/user-attachments/assets/db46609e-1b98-48d8-99c8-e166cfdfc6e3)
  </td>
    <td align="center">
      
  [speed_faster_output.webm](https://github.com/user-attachments/assets/0fbc14ca-dd4a-4362-aadc-afe0629f4c9f)
  </td>
  </tr>
  
</table>



```bash
# zero-shot cloning
# The path of the generated audio file is output/fear_zh_female_prompt_cloned.wav
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --tokenizer-path where_you_download_dir \
    --prompt-text "我总觉得，有人在跟着我，我能听到奇怪的脚步声。" \
    --prompt-audio "examples/fear_zh_female_prompt.wav" \
    --generated-text "可惜没有如果，已经发生的事情终究是发生了。" \
    --edit-type "clone" \
    --output-dir ./output 

python3 tts_infer.py \
    --model-path where_you_download_dir \
    --tokenizer-path where_you_download_dir \
    --prompt-text "His political stance was conservative, and he was particularly close to margaret thatcher." \
    --prompt-audio "examples/zero_shot_en_prompt.wav" \
    --generated-text "Underneath the courtyard is a large underground exhibition room which connects the two buildings.	" \
    --edit-type "clone" \
    --output-dir ./output 

# edit
# There will be one or multiple wave files corresponding to each edit iteration, for example: output/fear_zh_female_prompt_edited_iter1.wav, output/fear_zh_female_prompt_edited_iter2.wav, ...
# emotion; fear
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --tokenizer-path where_you_download_dir \
    --prompt-text "我总觉得，有人在跟着我，我能听到奇怪的脚步声。" \
    --prompt-audio "examples/fear_zh_female_prompt.wav" \
    --edit-type "emotion" \
    --edit-info "fear" \
    --output-dir ./output 

# emotion; happy
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --tokenizer-path where_you_download_dir \
    --prompt-text "You know, I just finished that big project and feel so relieved. Everything seems easier and more colorful, what a wonderful feeling!" \
    --prompt-audio "examples/en_happy_prompt.wav" \
    --edit-type "emotion" \
    --edit-info "happy" \
    --output-dir ./output 

# style; whisper
# for style whisper, the edit iteration num should be set bigger than 1 to get better results.
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --tokenizer-path where_you_download_dir \
    --prompt-text "比如在工作间隙，做一些简单的伸展运动，放松一下身体，这样，会让你更有精力." \
    --prompt-audio "examples/whisper_prompt.wav" \
    --edit-type "style" \
    --edit-info "whisper" \
    --output-dir ./output 

# paraliguistic 
# supported tags, Breathing, Laughter, Surprise-oh, Confirmation-en, Uhm, Surprise-ah, Surprise-wa, Sigh, Question-ei, Dissatisfaction-hnn
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --tokenizer-path where_you_download_dir \
    --prompt-text "我觉得这个计划大概是可行的，不过还需要再仔细考虑一下。" \
    --prompt-audio "examples/paralingustic_prompt.wav" \
    --generated-text "我觉得这个计划大概是可行的，[Uhm]不过还需要再仔细考虑一下。" \
    --edit-type "paralinguistic" \
    --output-dir ./output 

# denoise
# Prompt text is not needed.
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --tokenizer-path where_you_download_dir \
    --prompt-audio "examples/denoise_prompt.wav"\
    --edit-type "denoise" \
    --output-dir ./output 

# vad 
# Prompt text is not needed.
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --tokenizer-path where_you_download_dir \
    --prompt-audio "examples/vad_prompt.wav" \
    --edit-type "vad" \
    --output-dir ./output 

# speed
# supported edit-info: faster, slower, more faster, more slower
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --tokenizer-path where_you_download_dir \
    --prompt-text "上次你说鞋子有点磨脚，我给你买了一双软软的鞋垫。" \
    --prompt-audio "examples/speed_prompt.wav" \
    --edit-type "speed" \
    --edit-info "more faster" \
    --output-dir ./output 

```

