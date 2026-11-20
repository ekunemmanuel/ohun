"""Upload Ohùn's code to Kaggle and start a GPU run.

    uv run python kaggle/push.py code             # upload src/ohun as private dataset "ohun-code"
    uv run python kaggle/push.py run baselines    # start kaggle/baselines/run.py on a T4 x2
    uv run python kaggle/push.py run train-stt
    uv run python kaggle/push.py run train-stt --no-run   # create the notebook only (e.g. to add secrets)
    uv run python kaggle/push.py status train-stt
    uv run python kaggle/push.py fetch train-stt  # outputs -> runs/kaggle/train-stt

Needs `uvx --with six kaggle auth login` once. Everything is created private.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"  # gitignored
KAGGLE = ["uvx", "-q", "--with", "six", "kaggle"]


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    print("$ kaggle", " ".join(args), flush=True)
    result = subprocess.run([*KAGGLE, *args], text=True, capture_output=True)
    print(result.stdout.strip() or result.stderr.strip(), flush=True)
    if check and result.returncode != 0:
        raise SystemExit(f"kaggle {args[0]} {args[1]} failed")
    return result


def logged_in_user() -> str:
    view = subprocess.run([*KAGGLE, "config", "view"], text=True, capture_output=True).stdout
    for line in view.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("Not logged in to Kaggle: run `uvx --with six kaggle auth login`")


def push_code(user: str) -> None:
    out = BUILD / "ohun-code"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ROOT / "src" / "ohun", out / "ohun", ignore=shutil.ignore_patterns("__pycache__"))
    meta = {"title": "ohun-code", "id": f"{user}/ohun-code", "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    exists = kaggle("datasets", "status", f"{user}/ohun-code", check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(out), "-m", "update", "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(out), "--dir-mode", "zip")


def push_run(user: str, name: str, start: bool = True) -> None:
    out = BUILD / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    bootstrap = (HERE / "bootstrap.py").read_text()
    body = (HERE / name / "run.py").read_text().replace("from bootstrap import setup\n", "")
    code_file = f"ohun_{name.replace('-', '_')}.py"
    (out / code_file).write_text(f"# --- bootstrap.py ---\n{bootstrap}\n\n# --- {name}/run.py ---\n{body}")
    meta = {
        "id": f"{user}/ohun-{name}",
        "title": f"ohun-{name}",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{user}/ohun-code"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    kaggle("kernels", "push", "-p", str(out), "--accelerator", "NvidiaTeslaT4", *([] if start else ["--no-run"]))
    print(f"Watch it at https://www.kaggle.com/code/{user}/ohun-{name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", help="Kaggle username (default: the logged-in account)")
    p.add_argument("action", choices=["code", "run", "status", "fetch"])
    p.add_argument("name", nargs="?", choices=["baselines", "train-stt"])
    p.add_argument("--no-run", action="store_true", help="create/update the notebook without running it")
    a = p.parse_args()
    a.user = a.user or logged_in_user()
    if a.action == "code":
        push_code(a.user)
        return
    if not a.name:
        p.error(f"{a.action} needs a run name: baselines or train-stt")
    if a.action == "run":
        push_run(a.user, a.name, start=not a.no_run)
    elif a.action == "status":
        kaggle("kernels", "status", f"{a.user}/ohun-{a.name}")
    else:
        dest = ROOT / "runs" / "kaggle" / a.name
        dest.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", f"{a.user}/ohun-{a.name}", "-p", str(dest))


if __name__ == "__main__":
    main()
