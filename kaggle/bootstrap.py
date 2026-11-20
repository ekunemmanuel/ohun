"""Shared start of every Ohùn Kaggle run. kaggle/push.py copies it next to each run.py.

Finds the ohun code (uploaded as the private Kaggle dataset "ohun-code"), installs what the
Kaggle image lacks, logs in to Hugging Face from the Kaggle secret HF_TOKEN, and sends all
results to /kaggle/working (the part Kaggle keeps after the run).
"""

import glob
import os
import subprocess
import sys


def _check_internet() -> None:
    import socket

    try:
        socket.create_connection(("huggingface.co", 443), timeout=10).close()
    except OSError as err:
        sys.exit(
            "This Kaggle run has no internet, so it can't download models or data. Kaggle only "
            "gives internet to phone-verified accounts: verify at https://www.kaggle.com/settings "
            f"(Phone Verification), then run again. ({err})"
        )


def setup() -> None:
    _check_internet()
    # Kaggle unpacks the uploaded package flat (no ohun/ folder), and the mount path has
    # changed over time, so find train_stt.py anywhere and rebuild the package in /tmp.
    found = glob.glob("/kaggle/input/**/train_stt.py", recursive=True)
    if not found:
        listing = subprocess.run(["find", "/kaggle/input", "-maxdepth", "4"], capture_output=True, text=True).stdout
        sys.exit(f"ohun code not found under /kaggle/input. Is ohun-code attached? Saw:\n{listing}")
    import shutil

    shutil.copytree(os.path.dirname(found[0]), "/tmp/ohun-src/ohun", dirs_exist_ok=True)
    sys.path.insert(0, "/tmp/ohun-src")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "soundfile"], check=True)

    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as err:
        sys.exit(
            "No HF_TOKEN secret. In this notebook on kaggle.com: Add-ons > Secrets > add HF_TOKEN "
            f"(your Hugging Face read token) and switch it on, then run again. ({err})"
        )

    os.environ["OHUN_OUT"] = "/kaggle/working"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
