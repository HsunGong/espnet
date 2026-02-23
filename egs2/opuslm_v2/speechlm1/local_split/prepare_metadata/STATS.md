<!-- python local_split/prepare_metadata/stats_and_topk.py -i data/part2_4/metadata.jsonl -o data/part2_4/metadata.jsonl --type speech sound music  -->
=== Statistics (all valid records) ===
Total records: 2465417

Type         Count   MeanDur      0-5s     5-10s    10-15s    15-20s      20+s
------------------------------------------------------------------------------
speech     1976733     10.91    744289    621998     63044     89684    457718
music       283611      9.58      8366     37558    237687         0         0
sound       205073      8.67     21213     88392     95464         3         1

--- Per-dataset breakdown ---

  [speech]
    Dataset                                      Count   MeanDur
    ----------------------------------------  --------  --------
    owsm_finetune                              1715417      9.48
    owsm_v4_caption                             200184     23.39
    emilia_en                                    21398     12.53
    yodas_auto                                    8200     12.84
    laion_audio_300m_part1                        7406      7.35
    laion_audio_300m_part4                        7399      6.29
    laion_audio_300m_part3                        7245      8.08
    laion_audio_300m_part2                        7238      8.09
    yodas_manual                                  1450     11.18
    audioset                                       262     10.00
    youtube_8m_arkive                              220     10.00
    yt8m                                            94     10.00
    laion_disco_12m_part2                           65     10.00
    fma                                             63     10.00
    wavcaps                                         44      9.83
    laion_disco_12m_part1                           35     10.00
    laion_captioned_ai_music_snippets                9      8.63
    audiocaps                                        3      9.74
    mtg-jamendo-dataset                              1     10.00

  [music]
    Dataset                                      Count   MeanDur
    ----------------------------------------  --------  --------
    laion_disco_12m_part2                        92049      9.96
    laion_disco_12m_part1                        55438      9.96
    laion_captioned_ai_music_snippets            28199      9.20
    mtg-jamendo-dataset                          27878      9.98
    laion_audio_300m_part3                       17782      8.25
    laion_audio_300m_part2                       17594      8.29
    fma                                          16661      9.96
    laion_audio_300m_part1                       12284      8.26
    youtube_8m_arkive                             5864     10.00
    laion_audio_300m_part4                        3995      8.25
    audioset                                      2728     10.00
    yt8m                                          2533     10.00
    wavcaps                                        484      9.78
    laion_in_the_wild_sound_events                 114      7.02
    yodas_auto                                       5      5.30
    audiocaps                                        2     10.00
    yodas_manual                                     1     10.88

  [sound]
    Dataset                                      Count   MeanDur
    ----------------------------------------  --------  --------
    laion_audio_300m_part3                       56205      8.36
    laion_audio_300m_part2                       54281      8.34
    laion_audio_300m_part1                       39613      8.36
    wavcaps                                      23883      9.92
    laion_audio_300m_part4                       12348      8.23
    audioset                                      8275     10.00
    youtube_8m_arkive                             6731     10.00
    yt8m                                          2858     10.00
    clotho_train                                   344      9.59
    laion_in_the_wild_sound_events                 257      7.07
    audiocaps                                      156      9.94
    clotho_aqa                                      74      9.47
    yodas_auto                                      44      7.38
    yodas_manual                                     4      5.74

Input file:       data/part2_4/metadata.jsonl
Total lines:      2465417
Skipped bad json: 0
Skipped no path:  0
Skipped no dur:   0


<!-- sound -->
  sound [min_0.max_3]: 0  →  data/part2_4/metadata.sound.min_0.max_3.jsonl
  sound [min_3.max_4]: 9470  →  data/part2_4/metadata.sound.min_3.max_4.jsonl
  sound [min_4.max_5]: 11743  →  data/part2_4/metadata.sound.min_4.max_5.jsonl
  sound [min_5.max_6]: 13489  →  data/part2_4/metadata.sound.min_5.max_6.jsonl
  sound [min_6.max_7]: 15666  →  data/part2_4/metadata.sound.min_6.max_7.jsonl
  sound [min_7.max_8]: 17799  →  data/part2_4/metadata.sound.min_7.max_8.jsonl
  sound [min_8.max_9]: 19558  →  data/part2_4/metadata.sound.min_8.max_9.jsonl
  sound [min_9.max_10]: 21879  →  data/part2_4/metadata.sound.min_9.max_10.jsonl
  sound [min_10.max_11]: 69500  →  data/part2_4/metadata.sound.min_10.max_11.jsonl
  sound [min_11.max_12]: 25933  →  data/part2_4/metadata.sound.min_11.max_12.jsonl
  sound [min_12.max_13]: 28  →  data/part2_4/metadata.sound.min_12.max_13.jsonl
  sound [min_13.max_14]: 1  →  data/part2_4/metadata.sound.min_13.max_14.jsonl
  sound [min_14]: 5  →  data/part2_4/metadata.sound.min_14.jsonl

<!-- speech -->
 speech [min_0.max_3]: 210587  →  data/part2_4/metadata.speech.min_0.max_3.jsonl
  speech [min_3.max_4]: 253190  →  data/part2_4/metadata.speech.min_3.max_4.jsonl
  speech [min_4.max_5]: 280511  →  data/part2_4/metadata.speech.min_4.max_5.jsonl
  speech [min_5.max_6]: 260000  →  data/part2_4/metadata.speech.min_5.max_6.jsonl
  speech [min_6.max_7]: 183491  →  data/part2_4/metadata.speech.min_6.max_7.jsonl
  speech [min_7.max_8]: 102174  →  data/part2_4/metadata.speech.min_7.max_8.jsonl
  speech [min_8.max_9]: 50696  →  data/part2_4/metadata.speech.min_8.max_9.jsonl
  speech [min_9.max_10]: 25637  →  data/part2_4/metadata.speech.min_9.max_10.jsonl
  speech [min_10.max_11]: 17260  →  data/part2_4/metadata.speech.min_10.max_11.jsonl
  speech [min_11.max_12]: 13192  →  data/part2_4/metadata.speech.min_11.max_12.jsonl
  speech [min_12.max_13]: 9453  →  data/part2_4/metadata.speech.min_12.max_13.jsonl
  speech [min_13.max_14]: 10415  →  data/part2_4/metadata.speech.min_13.max_14.jsonl
  speech [min_14.max_15]: 12724  →  data/part2_4/metadata.speech.min_14.max_15.jsonl
  speech [min_15.max_16]: 14496  →  data/part2_4/metadata.speech.min_15.max_16.jsonl
  speech [min_16.max_17]: 16450  →  data/part2_4/metadata.speech.min_16.max_17.jsonl
  speech [min_17.max_18]: 18154  →  data/part2_4/metadata.speech.min_17.max_18.jsonl
  speech [min_18.max_19]: 19623  →  data/part2_4/metadata.speech.min_18.max_19.jsonl
  speech [min_19.max_20]: 20961  →  data/part2_4/metadata.speech.min_19.max_20.jsonl
  speech [min_20.max_21]: 7494  →  data/part2_4/metadata.speech.min_20.max_21.jsonl
  speech [min_21.max_22]: 9253  →  data/part2_4/metadata.speech.min_21.max_22.jsonl
  speech [min_22.max_23]: 12122  →  data/part2_4/metadata.speech.min_22.max_23.jsonl
  speech [min_23]: 428848  →  data/part2_4/metadata.speech.min_23.jsonl

<!-- music -->

 music [min_0.max_3]: 2  →  data/part2_4/metadata.music.min_0.max_3.jsonl
  music [min_3.max_4]: 3950  →  data/part2_4/metadata.music.min_3.max_4.jsonl
  music [min_4.max_5]: 4414  →  data/part2_4/metadata.music.min_4.max_5.jsonl
  music [min_5.max_6]: 6192  →  data/part2_4/metadata.music.min_5.max_6.jsonl
  music [min_6.max_7]: 6867  →  data/part2_4/metadata.music.min_6.max_7.jsonl
  music [min_7.max_8]: 7877  →  data/part2_4/metadata.music.min_7.max_8.jsonl
  music [min_8.max_9]: 8157  →  data/part2_4/metadata.music.min_8.max_9.jsonl
  music [min_9.max_10]: 8464  →  data/part2_4/metadata.music.min_9.max_10.jsonl
  music [min_10.max_11]: 229450  →  data/part2_4/metadata.music.min_10.max_11.jsonl
  music [min_11.max_12]: 8229  →  data/part2_4/metadata.music.min_11.max_12.jsonl
  music [min_12.max_13]: 7  →  data/part2_4/metadata.music.min_12.max_13.jsonl
  music [min_13.max_14]: 0  →  data/part2_4/metadata.music.min_13.max_14.jsonl
  music [min_14.max_15]: 0  →  data/part2_4/metadata.music.min_14.max_15.jsonl
  music [min_15.max_16]: 0  →  data/part2_4/metadata.music.min_15.max_16.jsonl
  music [min_16.max_17]: 0  →  data/part2_4/metadata.music.min_16.max_17.jsonl
  music [min_17.max_18]: 0  →  data/part2_4/metadata.music.min_17.max_18.jsonl
  music [min_18.max_19]: 0  →  data/part2_4/metadata.music.min_18.max_19.jsonl
  music [min_19.max_20]: 0  →  data/part2_4/metadata.music.min_19.max_20.jsonl
  music [min_20.max_21]: 0  →  data/part2_4/metadata.music.min_20.max_21.jsonl
  music [min_21.max_22]: 0  →  data/part2_4/metadata.music.min_21.max_22.jsonl
  music [min_22.max_23]: 0  →  data/part2_4/metadata.music.min_22.max_23.jsonl
  music [min_23]: 0  →  data/part2_4/metadata.music.min_23.jsonl

