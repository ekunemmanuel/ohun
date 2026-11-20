# Ohùn roadmap

Start small and measurable, then grow. Every phase ends with a number we can compare.

## Phase 0: foundation (done 2026-11-19)

Text normalization, scoring, dataset and model catalog with licenses. Tests pass.

## Phase 1: baselines, the numbers to beat

Before training anything, score what already exists on fixed test sets:

| | Yoruba | Pidgin |
|---|---|---|
| Test set | FLEURS `yo_ng` test | `pidgin-1.0` test (+ PidginMed for medical words) |
| Speech to text | Whisper small / turbo, MMS-1B, NCAIR Yoruba-ASR | Whisper small / turbo, MMS-1B |
| Text to speech | MMS-TTS yor | MMS-TTS pcm, Chatterbox Pidgin fine-tune |

Text to speech is scored two ways: **round trip** (synthesize the test sentences, transcribe
them with the best speech to text model, measure WER) and **listening** (native speakers
rate naturalness 1 to 5). The round trip is where the two halves of Ohùn check each other.

Output: `docs/BASELINES.md` (one row per run); every transcript in `runs/baselines/`.
Run one with `uv run --group eval ohun baseline --lang yor --model whisper-small`, or all
of them with `runs/run_baselines.sh`. Speech to text baselines started 2026-11-19; text to
speech baselines still to do.

## Phase 2: speech to text v1

Fine-tune **Whisper** (MIT) on Yoruba + Pidgin together: one model, two languages.
Start with `whisper-small` to learn cheaply, then `whisper-large-v3-turbo`.

- Pidgin data: WaZoBiaSpeech (1,017 h, CC-BY-4.0) + `pidgin-1.0` train.
- Yoruba data: FLEURS train + BibleTTS Yoruba (33 h, one speaker). Ship-safe Yoruba speech
  is thin (about 45 h); NaijaVoices has about 600 h but is non-commercial (decision pending). Keep only clips whose text
  has tone marks (`ohun.text.has_tone_marks`); untoned text would teach the model to drop them.
- Runs on Kaggle (T4 x2): `src/ohun/train_stt.py` via `kaggle/train-stt/run.py`. v1 is
  whisper-small, 4,000 steps of 16 clips, half Yoruba, half Pidgin.
- Success: beats every Phase 1 baseline on both test sets.

## Phase 3: text to speech v1

- Yoruba voice: BibleTTS Yoruba (33 h, one speaker, studio, fully toned, CC-BY-SA).
- Pidgin voice: `pidgin-1.0` (10 speakers, CC-BY-4.0). Not WaZoBiaSpeech: its card forbids
  text to speech and voice cloning. Pidgin voice data is the thinnest spot; Phase 4 matters here.
- Base: try Chatterbox (MIT) and Kokoro / StyleTTS2 (Apache), keep the better one.
- Yoruba gotcha: the model must read tone marks, so text goes in after `normalize_yoruba`.
- Success: beats MMS-TTS on round trip WER and in listening tests.

## Phase 4: our own data

Public Pidgin and toned-Yoruba data is thin. Recording our own (everyday Lagos Pidgin,
market and phone-call speech, numbers, names, code-switching with English) is where Ohùn
gets better than anyone else's model. Needs consent forms and a simple recording page.

## Phase 5: one model, more languages

- Add Hausa, Igbo and Nigerian English: an entry in `ohun.languages`, normalization rules,
  catalog data, then rerun Phases 1 to 3.
- Merge speech to text and text to speech into one model (a speech-language model that
  takes and produces both audio and text). This is research; we only start once v1 of both
  halves works and gives us a bar to beat.

## Phase 6: ship

An API and a demo page, then plug Ohùn into our own tools: ReadOut (read-aloud + dictation),
Cutroom (captions and voiceovers) and Fluency (reading coach).
