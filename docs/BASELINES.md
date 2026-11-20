# Baselines

Speech to text scores (Phase 1 baselines, then our own models). Lower is better. Yoruba WER keeps tone marks wherever the reference has them; "WER no tones" drops them. Speed is compute seconds per second of audio.

| Date | Lang | Model | Test set | Clips | WER | CER | WER no tones | Speed | Where |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-19 | pcm | whisper-small | nigerian-pidgin-1.0 test | 50 | 54.2% | 29.8% | - | 0.47x | Mac CPU |
| 2026-11-19 | yor | whisper-small | FLEURS yo_ng test | 50 | 122.9% | 83.8% | 119.5% | 0.85x | Mac CPU |
| 2026-11-19 | pcm | mms-asr | nigerian-pidgin-1.0 test | 50 | 50.6% | 22.9% | - | 2.30x | Mac CPU |
| 2026-11-19 | yor | mms-asr | FLEURS yo_ng test | 50 | 46.4% | 14.1% | 32.2% | 1.36x | Mac CPU |
| 2026-11-19 | pcm | whisper-turbo | nigerian-pidgin-1.0 test | 50 | 47.6% | 25.6% | - | 2.16x | Mac CPU |
| 2026-11-19 | yor | whisper-turbo | FLEURS yo_ng test | 50 | 106.4% | 44.3% | 94.1% | 1.70x | Mac CPU |
| 2026-11-19 | yor | whisper-small | FLEURS yo_ng test | 831 | 122.8% | 86.6% | 120.2% | 0.01x | GPU Tesla T4 |
| 2026-11-19 | pcm | whisper-small | nigerian-pidgin-1.0 test | 892 | 55.2% | 29.6% | - | 0.01x | GPU Tesla T4 |
| 2026-11-19 | pcm | whisper-small | WaZoBiaSpeech dev_test shard 0 | 377 | 46.1% | 27.6% | - | 0.01x | GPU Tesla T4 |
| 2026-11-19 | yor | whisper-turbo | FLEURS yo_ng test | 831 | 102.8% | 44.6% | 90.5% | 0.01x | GPU Tesla T4 |
| 2026-11-19 | pcm | whisper-turbo | nigerian-pidgin-1.0 test | 892 | 48.4% | 25.5% | - | 0.03x | GPU Tesla T4 |
| 2026-11-19 | pcm | whisper-turbo | WaZoBiaSpeech dev_test shard 0 | 377 | 39.3% | 22.2% | - | 0.02x | GPU Tesla T4 |
| 2026-11-19 | yor | mms-asr | FLEURS yo_ng test | 831 | 45.4% | 14.2% | 33.5% | 0.04x | GPU Tesla T4 |
| 2026-11-19 | pcm | mms-asr | nigerian-pidgin-1.0 test | 892 | 47.3% | 21.2% | - | 0.04x | GPU Tesla T4 |
| 2026-11-19 | pcm | mms-asr | WaZoBiaSpeech dev_test shard 0 | 377 | 37.9% | 15.7% | - | 0.04x | GPU Tesla T4 |
| 2026-11-19 | yor | ncair-yoruba-asr | FLEURS yo_ng test | 831 | 58.4% | 22.5% | 40.3% | 0.01x | GPU Tesla T4 |
| 2026-11-19 | yor | ohun-stt-small-v1 | FLEURS yo_ng test | 831 | 45.8% | 16.9% | 38.9% | 0.01x | GPU Tesla T4 |
| 2026-11-19 | pcm | ohun-stt-small-v1 | nigerian-pidgin-1.0 test | 892 | 21.1% | 10.7% | - | 0.01x | GPU Tesla T4 |
| 2026-11-19 | pcm | ohun-stt-small-v1 | WaZoBiaSpeech dev_test shard 0 | 377 | 18.6% | 12.0% | - | 0.01x | GPU Tesla T4 |

## What the full test sets show (Kaggle, 2026-11-19)

Rows marked GPU are the full test sets and are the ones to beat; Mac CPU rows are 50-clip samples.

| Test set | Best overall | Best we could ship on (MIT/Apache base) |
|---|---|---|
| Yoruba, FLEURS (831) | MMS 45.4% WER (33.5% without tones) | none usable: Whisper-turbo 102.8% |
| Pidgin, nigerian-pidgin-1.0 (892) | MMS 47.3% | Whisper-turbo 48.4% |
| Pidgin, WaZoBiaSpeech dev_test (377) | MMS 37.9% | Whisper-turbo 39.3% |

- MMS (non-commercial) is the strongest existing model in both languages.
- N-ATLaS Yoruba (NCAIR, Whisper-small fine-tune) scores 58.4%: about 1 in 5 of its outputs
  has no tone marks at all, and it struggles with numbers and names.
- Plain Whisper can't do Yoruba (loops and junk) and hears Pidgin as English
  ("don" -> "don't", "dey" -> "they", "na" -> "now").
- Ohùn v1 target: beat MMS on Pidgin and N-ATLaS on Yoruba with a model we can ship.

## Train/test overlap check (2026-11-20)

Checked on transcript text only (normalized with `ohun.text`), against `pidgin-1.0` train and a random
80-shard sample (31,964 clips, about 8%) of WaZoBiaSpeech train.

- `pidgin-1.0` test: 1 of 892 sentences also in `pidgin-1.0` train, 0 in the WaZoBia sample. Clean.
- WaZoBiaSpeech dev_test shard 0: **54 of 377 sentences (14%) appear exactly in the 8% sample of WaZoBia
  train**, 111 share a 6-word run. The share across the full train set is certainly higher. Speakers are
  disjoint (0 of 37 shared), so this is text leakage only. Treat the v1 WaZoBia score (18.6%) as inflated;
  v2 needs a de-duplicated WaZoBia test set (drop train sentences that appear in the test shard).
- `pidgin-1.0` test and train most likely share speakers (10 speakers in total). Not checked.

