"""Keep train text out of the test set: drop WaZoBiaSpeech dev_test clips whose sentence is in train.

v1 found 14% of dev_test shard 0 sentences word for word in just 8% of WaZoBia train, so that score
was inflated. Step 1 caches every train transcript (text only, never audio); step 2 writes the
dev_test clip ids whose normalized sentence is not in train. Both are resumable.

  uv run --group eval python -m ohun.dedupe cache      # slow: 839 shards, text columns only
  uv run --group eval python -m ohun.dedupe clean      # writes data/wazobia-dev-test-clean-ids.txt
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ohun.text import for_scoring

WAZOBIA = "Africanvoice/naija_african_voices"
DEV_TEST = "data/dev_test-00000-of-00046.parquet"  # the shard the baselines score
CACHE = Path("runs/cache/wazobia-train-text.jsonl")  # runs/ is not in git
CLEAN_IDS = Path("data/wazobia-dev-test-clean-ids.txt")


def _read(fs, name: str, cols: list[str]) -> dict[str, list]:
    import pyarrow.parquet as pq

    table = pq.ParquetFile(fs.open(f"datasets/{WAZOBIA}/{name}")).read(columns=cols)
    return {c: table.column(c).to_pylist() for c in cols}


def cache_train_text(workers: int = 4) -> None:
    from huggingface_hub import HfApi, HfFileSystem

    fs = HfFileSystem()
    names = sorted(f for f in HfApi().list_repo_files(WAZOBIA, repo_type="dataset") if f.startswith("data/train-"))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if CACHE.exists():
        done = {json.loads(line)["shard"] for line in CACHE.open() if line.strip()}
    todo = [n for n in names if n not in done]
    print(f"{len(done)} shards cached, {len(todo)} to go", flush=True)

    def one(name: str):
        for attempt in range(5):
            try:
                return name, _read(fs, name, ["transcript"])["transcript"]
            except Exception as e:  # network timeouts are common here
                print(f"retry {name}: {type(e).__name__}", flush=True)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"gave up on {name}")

    with ThreadPoolExecutor(workers) as ex, CACHE.open("a") as out:
        for i, (name, texts) in enumerate(ex.map(one, todo)):
            out.write(json.dumps({"shard": name, "text": texts}, ensure_ascii=False) + "\n")
            out.flush()
            if i % 20 == 0:
                print(f"{len(done) + i + 1}/{len(names)} shards", flush=True)


def clean_ids() -> list[str]:
    from huggingface_hub import HfFileSystem

    train = set()
    for line in CACHE.open():
        train.update(for_scoring(t, "pcm") for t in json.loads(line)["text"])
    dev = _read(HfFileSystem(), DEV_TEST, ["audio_id", "transcript"])
    keep = [i for i, t in zip(dev["audio_id"], dev["transcript"]) if for_scoring(t, "pcm") not in train]
    print(f"dev_test shard 0: kept {len(keep)} of {len(dev['audio_id'])} clips "
          f"({len(train)} distinct train sentences)", flush=True)
    return keep


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "cache":
        cache_train_text()
    elif cmd == "clean":
        CLEAN_IDS.write_text("\n".join(clean_ids()) + "\n")
        print(f"wrote {CLEAN_IDS}")
    else:
        sys.exit(__doc__)
