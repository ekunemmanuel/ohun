"""Ohùn: build the de-duplicated WaZoBiaSpeech dev_test id list. CPU only, text only.

Reads every train transcript (839 shards, text columns only), then writes
/kaggle/working/data/wazobia-dev-test-clean-ids.txt. Fetch it with
`uv run python kaggle/push.py fetch dedupe` and copy it to data/ in the repo.
The transcript cache (runs/cache/) comes back too, for de-duplicating v2 training data.
"""

from bootstrap import setup

setup()

import os  # noqa: E402

os.chdir("/kaggle/working")  # dedupe.py writes relative paths

from ohun import dedupe  # noqa: E402

dedupe.cache_train_text(workers=16)
dedupe.CLEAN_IDS.parent.mkdir(parents=True, exist_ok=True)
dedupe.CLEAN_IDS.write_text("\n".join(dedupe.clean_ids()) + "\n")
print(dedupe.CLEAN_IDS.read_text()[:200])
