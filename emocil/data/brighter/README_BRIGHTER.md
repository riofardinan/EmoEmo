---
language:
- af
- ar
- de
- en
- es
- ha
- hi
- id
- ig
- jv
- mr
- pcm
- pt
- ro
- ru
- rw
- su
- sv
- sw
- tt
- uk
- vmw
- xh
- yo
- zh
- zu
license: cc-by-4.0
dataset_info:
- config_name: afr
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 257405
    num_examples: 1222
  - name: dev
    num_bytes: 37576
    num_examples: 196
  - name: test
    num_bytes: 443683
    num_examples: 2130
  download_size: 185339
  dataset_size: 738664
- config_name: arq
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 189789
    num_examples: 901
  - name: dev
    num_bytes: 38343
    num_examples: 200
  - name: test
    num_bytes: 375777
    num_examples: 1804
  download_size: 180591
  dataset_size: 603909
- config_name: ary
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 338714
    num_examples: 1608
  - name: dev
    num_bytes: 102376
    num_examples: 534
  - name: test
    num_bytes: 338282
    num_examples: 1624
  download_size: 267222
  dataset_size: 779372
- config_name: chn
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 556519
    num_examples: 2642
  - name: dev
    num_bytes: 76686
    num_examples: 400
  - name: test
    num_bytes: 1100668
    num_examples: 5284
  download_size: 789776
  dataset_size: 1733873
- config_name: deu
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 548303
    num_examples: 2603
  - name: dev
    num_bytes: 76686
    num_examples: 400
  - name: test
    num_bytes: 1084837
    num_examples: 5208
  download_size: 894088
  dataset_size: 1709826
- config_name: eng
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 582217
    num_examples: 2764
  - name: dev
    num_bytes: 44094
    num_examples: 230
  - name: test
    num_bytes: 1151494
    num_examples: 5528
  download_size: 402292
  dataset_size: 1777805
- config_name: esp
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 420443
    num_examples: 1996
  - name: dev
    num_bytes: 70551
    num_examples: 368
  - name: test
    num_bytes: 706144
    num_examples: 3390
  download_size: 216719
  dataset_size: 1197138
- config_name: hau
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 451829
    num_examples: 2145
  - name: dev
    num_bytes: 136501
    num_examples: 712
  - name: test
    num_bytes: 449932
    num_examples: 2160
  download_size: 258761
  dataset_size: 1038262
- config_name: hin
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 538403
    num_examples: 2556
  - name: dev
    num_bytes: 38343
    num_examples: 200
  - name: test
    num_bytes: 420770
    num_examples: 2020
  download_size: 377557
  dataset_size: 997516
- config_name: ibo
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 606652
    num_examples: 2880
  - name: dev
    num_bytes: 183663
    num_examples: 958
  - name: test
    num_bytes: 601576
    num_examples: 2888
  download_size: 266394
  dataset_size: 1391891
- config_name: ind
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: dev
    num_bytes: 29907
    num_examples: 156
  - name: test
    num_bytes: 177265
    num_examples: 851
  download_size: 77485
  dataset_size: 207172
- config_name: jav
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: dev
    num_bytes: 28949
    num_examples: 151
  - name: test
    num_bytes: 174348
    num_examples: 837
  download_size: 83743
  dataset_size: 203297
- config_name: kin
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 516286
    num_examples: 2451
  - name: dev
    num_bytes: 156056
    num_examples: 814
  - name: test
    num_bytes: 512839
    num_examples: 2462
  download_size: 430821
  dataset_size: 1185181
- config_name: mar
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 508703
    num_examples: 2415
  - name: dev
    num_bytes: 38343
    num_examples: 200
  - name: test
    num_bytes: 416604
    num_examples: 2000
  download_size: 348097
  dataset_size: 963650
- config_name: pcm
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 785277
    num_examples: 3728
  - name: dev
    num_bytes: 237727
    num_examples: 1240
  - name: test
    num_bytes: 779050
    num_examples: 3740
  download_size: 574462
  dataset_size: 1802054
- config_name: ptbr
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 468891
    num_examples: 2226
  - name: dev
    num_bytes: 76686
    num_examples: 400
  - name: test
    num_bytes: 927361
    num_examples: 4452
  download_size: 455209
  dataset_size: 1472938
- config_name: ptmz
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 325654
    num_examples: 1546
  - name: dev
    num_bytes: 98541
    num_examples: 514
  - name: test
    num_bytes: 323284
    num_examples: 1552
  download_size: 166765
  dataset_size: 747479
- config_name: ron
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 261408
    num_examples: 1241
  - name: dev
    num_bytes: 47162
    num_examples: 246
  - name: test
    num_bytes: 466180
    num_examples: 2238
  download_size: 225486
  dataset_size: 774750
- config_name: rus
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 564312
    num_examples: 2679
  - name: dev
    num_bytes: 76302
    num_examples: 398
  - name: test
    num_bytes: 416604
    num_examples: 2000
  download_size: 312212
  dataset_size: 1057218
- config_name: sun
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 194634
    num_examples: 924
  - name: dev
    num_bytes: 76302
    num_examples: 398
  - name: test
    num_bytes: 385775
    num_examples: 1852
  download_size: 152346
  dataset_size: 656711
- config_name: swa
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 696596
    num_examples: 3307
  - name: dev
    num_bytes: 211270
    num_examples: 1102
  - name: test
    num_bytes: 689896
    num_examples: 3312
  download_size: 409881
  dataset_size: 1597762
- config_name: swe
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 250033
    num_examples: 1187
  - name: dev
    num_bytes: 76686
    num_examples: 400
  - name: test
    num_bytes: 494925
    num_examples: 2376
  download_size: 375901
  dataset_size: 821644
- config_name: tat
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 210643
    num_examples: 1000
  - name: dev
    num_bytes: 76686
    num_examples: 400
  - name: test
    num_bytes: 416604
    num_examples: 2000
  download_size: 186687
  dataset_size: 703933
- config_name: ukr
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 519445
    num_examples: 2466
  - name: dev
    num_bytes: 95474
    num_examples: 498
  - name: test
    num_bytes: 930693
    num_examples: 4468
  download_size: 393469
  dataset_size: 1545612
- config_name: vmw
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 326707
    num_examples: 1551
  - name: dev
    num_bytes: 98925
    num_examples: 516
  - name: test
    num_bytes: 323701
    num_examples: 1554
  download_size: 191579
  dataset_size: 749333
- config_name: xho
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: dev
    num_bytes: 130750
    num_examples: 682
  - name: test
    num_bytes: 332033
    num_examples: 1594
  download_size: 113314
  dataset_size: 462783
- config_name: yor
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: train
    num_bytes: 630244
    num_examples: 2992
  - name: dev
    num_bytes: 190565
    num_examples: 994
  - name: test
    num_bytes: 624906
    num_examples: 3000
  download_size: 400802
  dataset_size: 1445715
- config_name: zul
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: anger
    dtype: int64
  - name: disgust
    dtype: int64
  - name: fear
    dtype: int64
  - name: joy
    dtype: int64
  - name: sadness
    dtype: int64
  - name: surprise
    dtype: int64
  - name: emotions
    list: string
  splits:
  - name: dev
    num_bytes: 167751
    num_examples: 875
  - name: test
    num_bytes: 426394
    num_examples: 2047
  download_size: 156608
  dataset_size: 594145
configs:
- config_name: afr
  data_files:
  - split: train
    path: afr/train-*
  - split: dev
    path: afr/dev-*
  - split: test
    path: afr/test-*
- config_name: arq
  data_files:
  - split: train
    path: arq/train-*
  - split: dev
    path: arq/dev-*
  - split: test
    path: arq/test-*
- config_name: ary
  data_files:
  - split: train
    path: ary/train-*
  - split: dev
    path: ary/dev-*
  - split: test
    path: ary/test-*
- config_name: chn
  data_files:
  - split: train
    path: chn/train-*
  - split: dev
    path: chn/dev-*
  - split: test
    path: chn/test-*
- config_name: deu
  data_files:
  - split: train
    path: deu/train-*
  - split: dev
    path: deu/dev-*
  - split: test
    path: deu/test-*
- config_name: eng
  data_files:
  - split: train
    path: eng/train-*
  - split: dev
    path: eng/dev-*
  - split: test
    path: eng/test-*
- config_name: esp
  data_files:
  - split: train
    path: esp/train-*
  - split: dev
    path: esp/dev-*
  - split: test
    path: esp/test-*
- config_name: hau
  data_files:
  - split: train
    path: hau/train-*
  - split: dev
    path: hau/dev-*
  - split: test
    path: hau/test-*
- config_name: hin
  data_files:
  - split: train
    path: hin/train-*
  - split: dev
    path: hin/dev-*
  - split: test
    path: hin/test-*
- config_name: ibo
  data_files:
  - split: train
    path: ibo/train-*
  - split: dev
    path: ibo/dev-*
  - split: test
    path: ibo/test-*
- config_name: ind
  data_files:
  - split: dev
    path: ind/dev-*
  - split: test
    path: ind/test-*
- config_name: jav
  data_files:
  - split: dev
    path: jav/dev-*
  - split: test
    path: jav/test-*
- config_name: kin
  data_files:
  - split: train
    path: kin/train-*
  - split: dev
    path: kin/dev-*
  - split: test
    path: kin/test-*
- config_name: mar
  data_files:
  - split: train
    path: mar/train-*
  - split: dev
    path: mar/dev-*
  - split: test
    path: mar/test-*
- config_name: pcm
  data_files:
  - split: train
    path: pcm/train-*
  - split: dev
    path: pcm/dev-*
  - split: test
    path: pcm/test-*
- config_name: ptbr
  data_files:
  - split: train
    path: ptbr/train-*
  - split: dev
    path: ptbr/dev-*
  - split: test
    path: ptbr/test-*
- config_name: ptmz
  data_files:
  - split: train
    path: ptmz/train-*
  - split: dev
    path: ptmz/dev-*
  - split: test
    path: ptmz/test-*
- config_name: ron
  data_files:
  - split: train
    path: ron/train-*
  - split: dev
    path: ron/dev-*
  - split: test
    path: ron/test-*
- config_name: rus
  data_files:
  - split: train
    path: rus/train-*
  - split: dev
    path: rus/dev-*
  - split: test
    path: rus/test-*
- config_name: sun
  data_files:
  - split: train
    path: sun/train-*
  - split: dev
    path: sun/dev-*
  - split: test
    path: sun/test-*
- config_name: swa
  data_files:
  - split: train
    path: swa/train-*
  - split: dev
    path: swa/dev-*
  - split: test
    path: swa/test-*
- config_name: swe
  data_files:
  - split: train
    path: swe/train-*
  - split: dev
    path: swe/dev-*
  - split: test
    path: swe/test-*
- config_name: tat
  data_files:
  - split: train
    path: tat/train-*
  - split: dev
    path: tat/dev-*
  - split: test
    path: tat/test-*
- config_name: ukr
  data_files:
  - split: train
    path: ukr/train-*
  - split: dev
    path: ukr/dev-*
  - split: test
    path: ukr/test-*
- config_name: vmw
  data_files:
  - split: train
    path: vmw/train-*
  - split: dev
    path: vmw/dev-*
  - split: test
    path: vmw/test-*
- config_name: xho
  data_files:
  - split: dev
    path: xho/dev-*
  - split: test
    path: xho/test-*
- config_name: yor
  data_files:
  - split: train
    path: yor/train-*
  - split: dev
    path: yor/dev-*
  - split: test
    path: yor/test-*
- config_name: zul
  data_files:
  - split: dev
    path: zul/dev-*
  - split: test
    path: zul/test-*
---

# BRIGHTER Emotion Categories Dataset

This dataset contains the emotion categories data from the BRIGHTER paper: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages.

## Dataset Description

The BRIGHTER Emotion Categories dataset is a comprehensive multi-language, multi-label emotion classification dataset with separate configurations for each language. It represents one of the largest human-annotated emotion datasets across multiple languages.

- **Total languages**: 28 languages
- **Total examples**: 139583
- **Splits**: train, dev, test

## About BRIGHTER

BRIGHTER addresses the gap in human-annotated textual emotion recognition datasets for low-resource languages. While most existing emotion datasets focus on English, BRIGHTER covers multiple languages, including many low-resource ones. The dataset was created by selecting text from various sources and having annotators label six categorical emotions: anger, disgust, fear, joy, sadness, and surprise.

The dataset contains text in the following languages: Afrikaans, Algerian Arabic, Moroccan Arabic, Mandarin Chinese, German, English, Spanish (Ecuador, Colombia, Mexico), Hausa, Hindi, Igbo, Indonesian, Javanese, Kinyarwanda, Marathi, Nigerian Pidgin, Portuguese (Brazil), Portuguese (Mozambique), Romanian, Russian, Sundanese, Swahili, Swedish, Tatar, Ukrainian, Makhuwa, Xhosa, Yoruba, and Zulu.

## Language Configurations

Each language is available as a separate configuration with the following statistics:

| Original Code | ISO Code | Train Examples | Dev Examples | Test Examples | Total |
|---------------|----------|---------------|-------------|--------------|-------|
| afr | af | 1222 | 196 | 2130 | 3548 |
| arq | ar | 901 | 200 | 1804 | 2905 |
| ary | ar | 1608 | 534 | 1624 | 3766 |
| chn | zh | 2642 | 400 | 5284 | 8326 |
| deu | de | 2603 | 400 | 5208 | 8211 |
| eng | en | 2764 | 230 | 5528 | 8522 |
| esp | es | 1996 | 368 | 3390 | 5754 |
| hau | ha | 2145 | 712 | 2160 | 5017 |
| hin | hi | 2556 | 200 | 2020 | 4776 |
| ibo | ig | 2880 | 958 | 2888 | 6726 |
| ind | id | 0 | 156 | 851 | 1007 |
| jav | jv | 0 | 151 | 837 | 988 |
| kin | rw | 2451 | 814 | 2462 | 5727 |
| mar | mr | 2415 | 200 | 2000 | 4615 |
| pcm | pcm | 3728 | 1240 | 3740 | 8708 |
| ptbr | pt | 2226 | 400 | 4452 | 7078 |
| ptmz | pt | 1546 | 514 | 1552 | 3612 |
| ron | ro | 1241 | 246 | 2238 | 3725 |
| rus | ru | 2679 | 398 | 2000 | 5077 |
| sun | su | 924 | 398 | 1852 | 3174 |
| swa | sw | 3307 | 1102 | 3312 | 7721 |
| swe | sv | 1187 | 400 | 2376 | 3963 |
| tat | tt | 1000 | 400 | 2000 | 3400 |
| ukr | uk | 2466 | 498 | 4468 | 7432 |
| vmw | vmw | 1551 | 516 | 1554 | 3621 |
| xho | xh | 0 | 682 | 1594 | 2276 |
| yor | yo | 2992 | 994 | 3000 | 6986 |
| zul | zu | 0 | 875 | 2047 | 2922 |

## Features

- **id**: Unique identifier for each example
- **text**: Text content to classify
- **anger**, **disgust**, **fear**, **joy**, **sadness**, **surprise**: Presence of emotion
- **emotions**: List of emotions present in the text

## Dataset Characteristics

This dataset provides binary labels for emotion presence, making it suitable for multi-label classification tasks. For regression tasks or fine-grained emotion analysis, please see the companion BRIGHTER-emotion-intensities dataset.

## Usage

```python
from datasets import load_dataset

# Load all data for a specific language
eng_dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "eng")

# Or load a specific split for a language
eng_train = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "eng", split="train")
```

## Citation

If you use this dataset, please cite the following papers:

```
@misc{muhammad2025brighterbridginggaphumanannotated,
      title={BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages}, 
      author={Shamsuddeen Hassan Muhammad and Nedjma Ousidhoum and Idris Abdulmumin and Jan Philip Wahle and Terry Ruas and Meriem Beloucif and Christine de Kock and Nirmal Surange and Daniela Teodorescu and Ibrahim Said Ahmad and David Ifeoluwa Adelani and Alham Fikri Aji and Felermino D. M. A. Ali and Ilseyar Alimova and Vladimir Araujo and Nikolay Babakov and Naomi Baes and Ana-Maria Bucur and Andiswa Bukula and Guanqun Cao and Rodrigo Tufiño and Rendi Chevi and Chiamaka Ijeoma Chukwuneke and Alexandra Ciobotaru and Daryna Dementieva and Murja Sani Gadanya and Robert Geislinger and Bela Gipp and Oumaima Hourrane and Oana Ignat and Falalu Ibrahim Lawan and Rooweither Mabuya and Rahmad Mahendra and Vukosi Marivate and Andrew Piper and Alexander Panchenko and Charles Henrique Porto Ferreira and Vitaly Protasov and Samuel Rutunda and Manish Shrivastava and Aura Cristina Udrea and Lilian Diana Awuor Wanzare and Sophie Wu and Florian Valentin Wunderlich and Hanif Muhammad Zhafran and Tianhui Zhang and Yi Zhou and Saif M. Mohammad},
      year={2025},
      eprint={2502.11926},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11926}, 
}
```

```
@misc{muhammad2025semeval2025task11bridging,
      title={SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection}, 
      author={Shamsuddeen Hassan Muhammad and Nedjma Ousidhoum and Idris Abdulmumin and Seid Muhie Yimam and Jan Philip Wahle and Terry Ruas and Meriem Beloucif and Christine De Kock and Tadesse Destaw Belay and Ibrahim Said Ahmad and Nirmal Surange and Daniela Teodorescu and David Ifeoluwa Adelani and Alham Fikri Aji and Felermino Ali and Vladimir Araujo and Abinew Ali Ayele and Oana Ignat and Alexander Panchenko and Yi Zhou and Saif M. Mohammad},
      year={2025},
      eprint={2503.07269},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.07269}, 
}
```

## License
This dataset is licensed under CC-BY 4.0.
