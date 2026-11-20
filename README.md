# Ohùn

*Ohùn* is Yoruba for "voice".

Speech to text and text to speech for Nigerian languages. We start with **Yoruba** and
**Nigerian Pidgin**, then add Hausa, Igbo and Nigerian English.

The goal is our own models, not a wrapper around someone else's API: one speech to text
model that hears Yoruba and Pidgin, and text to speech voices that speak them, built on
open bases we're allowed to ship. Later they merge into a single model (see
[docs/ROADMAP.md](docs/ROADMAP.md)).

## Where it stands

Phase 0, the foundation:

- `ohun.text`: Yoruba and Pidgin text normalization. Yoruba tone marks and under-dots are
  kept and unified (precomposed vs decomposed, vertical line vs dot below), because
  ọkọ́ / ọkọ̀ / ọkọ (hoe / vehicle / husband) are different words.
- `ohun.metrics`: WER and CER. For Yoruba it also reports WER without tones and without any
  marks, so we can see how much error is tone alone.
- `data/catalog.toml`: every dataset and base model we know of, with hours, license and
  whether it's **ok to ship** (commercial) or **research only**.

## Use

Needs [uv](https://docs.astral.sh/uv/) (installed through mise on this Mac).

```sh
uv sync
uv run pytest                                   # 19 tests
uv run ohun catalog                             # everything, with licenses
uv run ohun catalog --lang pcm --task stt       # Pidgin speech to text data and bases
uv run ohun catalog --commercial                # only what we can ship
uv run ohun normalize --lang yor --level no_tone "Ọkọ̀ náà ti dé"
uv run ohun score --lang yor refs.txt hyps.txt  # one sentence per line

# Phase 1 baselines (PyTorch etc. live in the "eval" group; needs `hf auth login`)
uv run --group eval ohun baseline --lang pcm --model whisper-small --n 50
```

Results so far: [docs/BASELINES.md](docs/BASELINES.md).

## Kaggle (GPU)

Training and full-test-set scoring run on Kaggle's free T4 GPUs. One-time setup:
`uvx --with six kaggle auth login`, then on kaggle.com add your Hugging Face read token as
the secret `HF_TOKEN` on each Ohùn notebook (Add-ons > Secrets).

```sh
uv run python kaggle/push.py code              # upload src/ohun (private dataset)
uv run python kaggle/push.py run baselines     # every baseline, full test sets
uv run python kaggle/push.py run train-stt     # fine-tune Whisper (about 3-4 h)
uv run python kaggle/push.py status train-stt
uv run python kaggle/push.py fetch train-stt   # results + model -> runs/kaggle/
```

Before pushing training changes, run the 2-step smoke test on the Mac:
`uv run --group eval python -c "from ohun.train_stt import Config, train; train(Config.smoke_test())"`.

## Hardware

This Mac (Intel i5, 16 GB, no NVIDIA GPU) handles data prep, scoring and small model
runs. PyTorch's last Intel Mac build is 2.2.2, which is why Python is pinned to 3.12.
Fine-tuning runs on a rented or free cloud GPU (Kaggle, Colab, Modal or RunPod).

## Rules

- License discipline: only `commercial = true` data goes into the model we ship.
  Research-only sets are for comparison and evaluation.
- Never commit audio, datasets, checkpoints or keys (see `.gitignore`).
- Re-check a dataset's license before training on it: owners change terms.
