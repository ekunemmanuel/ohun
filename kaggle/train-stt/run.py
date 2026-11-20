"""Ohùn Phase 2 on a Kaggle GPU: fine-tune Whisper on Yoruba + Pidgin, then score it.

The trained model lands in /kaggle/working/ohun-stt (safetensors). Fetch it with
`kaggle kernels output <user>/ohun-train-stt -p runs/kaggle/train-stt`.
"""

from bootstrap import setup

setup()

from ohun.train_stt import Config, evaluate_trained, train  # noqa: E402

cfg = Config()
print(cfg, flush=True)
model_dir = train(cfg)
evaluate_trained(model_dir, "ohun-stt-small-v1")
print(open("/kaggle/working/docs/BASELINES.md").read())
