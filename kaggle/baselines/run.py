"""Ohùn Phase 1 on a Kaggle GPU: every baseline model on the full test sets."""

from bootstrap import setup

setup()

from ohun.baseline import MODELS, TEST_SETS, evaluate  # noqa: E402

for model_id, spec in MODELS.items():
    for test_key, test in TEST_SETS.items():
        if test.lang in spec.langs:
            try:
                evaluate(test_key, model_id, spec)
            except Exception as err:  # keep going: one broken model shouldn't cost the others
                print(f"FAILED {model_id} on {test_key}: {err!r}", flush=True)

print(open("/kaggle/working/docs/BASELINES.md").read())
