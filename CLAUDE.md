# Ohùn

Our own speech to text and text to speech models for Nigerian languages. Yoruba (`yor`)
and Nigerian Pidgin (`pcm`) first, then Hausa, Igbo, Nigerian English. README.md says what
exists; docs/ROADMAP.md says what's next and where we are.

- Python 3.12 via uv (`uv sync`, `uv run pytest`, `uv run ohun ...`). The 3.12 pin is on
  purpose: this is an Intel Mac and PyTorch 2.2.2 is its last build. Training happens on a
  cloud GPU, never assume CUDA locally.
- Languages are ISO 639-3 codes everywhere. New language: `src/ohun/languages.py`, then
  normalization in `src/ohun/text.py`, then catalog entries.
- Yoruba tone marks and under-dots carry meaning. Never strip them from training text or
  the headline score. All text for training or scoring goes through `ohun.text`.
- `data/catalog.toml` is the single list of datasets and base models. Only
  `commercial = true` entries may go into a model we ship; the rest are for evaluation.
  Record the license and the date you checked it when adding an entry.
- Never commit audio, datasets, checkpoints, model weights or `.env`.
- GPU work goes through `kaggle/push.py` (see README). Kaggle runs one file per script, so
  `push.py` glues `kaggle/bootstrap.py` onto each `kaggle/<name>/run.py`. Smoke-test training
  changes on the Mac (`Config.smoke_test()`) before spending Kaggle GPU hours.
- Local ML deps are the `eval` group, pinned for the Intel Mac: torch 2.2.2, numpy<2,
  transformers 4.x, pyarrow<22. Models shipped only as pytorch_model.bin (NCAIR) need torch
  >= 2.6, so they are scored on Kaggle.
- WaZoBiaSpeech's card forbids text to speech / voice cloning use: speech to text only.
- Run `uv run pytest` before finishing.
