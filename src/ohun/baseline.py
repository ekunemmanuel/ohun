"""Score speech to text models on fixed test sets (Phase 1 baselines, and every model we train).

Locally (CPU):  uv run --group eval ohun baseline --lang yor --model whisper-small --n 50
On Kaggle (GPU): kaggle/baselines/run.py calls evaluate() on the full test sets.

Clips are a fixed random sample (seed 0), so every model hears exactly the same audio.
Each run writes every transcript to runs/baselines/<test>-<model>-n<clips>.jsonl and appends
a row to docs/BASELINES.md, both under OHUN_OUT (default: the repo).
"""

import io
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("OHUN_OUT", ROOT))
RUNS = OUT / "runs" / "baselines"
RESULTS = OUT / "docs" / "BASELINES.md"
SAMPLE_RATE = 16_000
HEADER = (
    "# Baselines\n\nSpeech to text scores (Phase 1 baselines, then our own models). Lower is better. "
    "Yoruba WER keeps tone marks wherever the reference has them; \"WER no tones\" drops them. "
    "Speed is compute seconds per second of audio.\n\n"
    "| Date | Lang | Model | Test set | Clips | WER | CER | WER no tones | Speed | Where |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


@dataclass(frozen=True)
class TestSet:
    key: str
    lang: str
    repo: str
    file: str
    text_column: str
    id_column: str
    name: str
    audio_column: str = "audio"


TEST_SETS = {
    t.key: t
    for t in [
        TestSet(
            "fleurs-yor", "yor", "google/fleurs", "parquet-data/yo_ng/test-00000-of-00001.parquet",
            "raw_transcription", "id", "FLEURS yo_ng test",
        ),
        TestSet(
            "pidgin-1.0", "pcm", "asr-nigerian-pidgin/nigerian-pidgin-1.0",
            "data/test-00000-of-00001-16c049b6814bb7ff.parquet", "sentence", "filename", "nigerian-pidgin-1.0 test",
        ),
        # Speaker-disjoint from WaZoBiaSpeech train. One shard, so a handful of speakers only.
        TestSet(
            "wazobia-dev-test", "pcm", "Africanvoice/naija_african_voices",
            "data/dev_test-00000-of-00046.parquet", "transcript", "audio_id", "WaZoBiaSpeech dev_test shard 0",
            audio_column="audio_path",
        ),
    ]
}
DEFAULT_TEST = {"yor": "fleurs-yor", "pcm": "pidgin-1.0"}


@dataclass(frozen=True)
class ModelSpec:
    repo: str  # Hugging Face repo id, or a local folder for a model we trained
    kind: str  # "whisper" or "mms"
    langs: tuple[str, ...] = ("yor", "pcm")


# Keys match ids in data/catalog.toml.
MODELS = {
    "whisper-small": ModelSpec("openai/whisper-small", "whisper"),
    "whisper-turbo": ModelSpec("openai/whisper-large-v3-turbo", "whisper"),
    "mms-asr": ModelSpec("facebook/mms-1b-all", "mms"),
    # Ships pytorch_model.bin, which transformers only loads with torch >= 2.6: Kaggle, not the Mac.
    "ncair-yoruba-asr": ModelSpec("NCAIR1/Yoruba-ASR", "whisper", ("yor",)),
}


def decode_audio(data: bytes):
    """Any wav/flac bytes -> 16 kHz mono float32."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype("float32")
    return audio


def load_clips(test_key: str, n: int | None = None, seed: int = 0) -> list[dict]:
    """n=None loads the whole test set."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    test = TEST_SETS[test_key]
    path = hf_hub_download(test.repo, test.file, repo_type="dataset")
    table = pq.read_table(path, columns=[test.id_column, test.text_column, test.audio_column])
    rows = range(table.num_rows)
    picks = rows if n is None or n >= table.num_rows else sorted(random.Random(seed).sample(rows, n))
    clips = []
    for i in picks:
        row = table.slice(i, 1).to_pylist()[0]
        if not (row[test.text_column] or "").strip():
            continue
        audio = decode_audio(row[test.audio_column]["bytes"])
        clips.append({"id": str(row[test.id_column]), "text": row[test.text_column], "audio": audio})
    return clips


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _whisper(repo: str, lang: str, device: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from ohun import languages

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(repo)
    model = WhisperForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
    language = languages.get(lang).whisper

    def transcribe(batch: list) -> list[str]:
        inputs = processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            ids = model.generate(
                inputs.input_features.to(device, dtype),
                attention_mask=inputs.attention_mask.to(device),
                language=language,
                task="transcribe",
                max_new_tokens=220,
            )
        return processor.batch_decode(ids, skip_special_tokens=True)

    return transcribe, (16 if device == "cuda" else 4)


def _mms(repo: str, lang: str, device: str):
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(repo, target_lang=lang)
    model = Wav2Vec2ForCTC.from_pretrained(repo, target_lang=lang, ignore_mismatched_sizes=True).to(device).eval()

    def transcribe(batch: list) -> list[str]:
        inputs = processor(batch[0], sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        return [processor.decode(logits[0].argmax(dim=-1))]

    return transcribe, 1


def evaluate(test_key: str, model_name: str, spec: ModelSpec, n: int | None = None, where: str = ""):
    """Transcribe a test set with one model, save transcripts, append a results row. Returns the Score."""
    import torch

    from ohun.metrics import score

    test = TEST_SETS[test_key]
    if test.lang not in spec.langs:
        raise ValueError(f"{model_name} doesn't cover {test.lang}")
    device = _device()
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    where = where or ("GPU " + torch.cuda.get_device_name(0) if device == "cuda" else "Mac CPU")

    print(f"Loading {'all' if n is None else n} clips of {test.name} ...", flush=True)
    clips = load_clips(test_key, n)
    print(f"Loading {spec.repo} on {device} ...", flush=True)
    transcribe, batch_size = (_whisper if spec.kind == "whisper" else _mms)(spec.repo, test.lang, device)

    hyps: list[str] = []
    audio_seconds = sum(len(c["audio"]) for c in clips) / SAMPLE_RATE
    start = time.perf_counter()
    for i in range(0, len(clips), batch_size):
        hyps += [h.strip() for h in transcribe([c["audio"] for c in clips[i : i + batch_size]])]
        print(f"  {len(hyps)}/{len(clips)}", flush=True)
    speed = (time.perf_counter() - start) / audio_seconds

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{test_key}-{model_name}-n{len(clips)}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c, h in zip(clips, hyps):
            f.write(json.dumps({"id": c["id"], "ref": c["text"], "hyp": h}, ensure_ascii=False) + "\n")

    result = score([c["text"] for c in clips], hyps, test.lang)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS.exists():
        RESULTS.write_text(HEADER, encoding="utf-8")
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(_row(str(date.today()), test, model_name, result, f"{speed:.2f}x", where) + "\n")
    print(result.summary(), f"| {speed:.2f}s per audio second on {where}", f"| transcripts: {out}", flush=True)
    return result


def run(lang: str, model_id: str, n: int | None = 50, test_key: str | None = None):
    return evaluate(test_key or DEFAULT_TEST[lang], model_id, MODELS[model_id], n)


def _row(run_date: str, test: TestSet, model_name: str, s, speed: str, where: str) -> str:
    no_tone = f"{s.wer_no_tone:.1%}" if s.wer_no_tone is not None else "-"
    return (
        f"| {run_date} | {test.lang} | {model_name} | {test.name} | {s.n} | "
        f"{s.wer:.1%} | {s.cer:.1%} | {no_tone} | {speed} | {where} |"
    )


def rescore() -> None:
    """Recompute WER/CER in docs/BASELINES.md from the saved transcripts.

    Run after changing normalization or scoring: no model is rerun; date, speed and place are kept.
    """
    from ohun.metrics import score

    by_name = {t.name: t for t in TEST_SETS.values()}
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 10 or cells[3] not in by_name:
            continue
        run_date, _, model_name, test_name, clips, *_, speed, where = cells
        test = by_name[test_name]
        path = RUNS / f"{test.key}-{model_name}-n{clips}.jsonl"
        if not path.exists():
            print(f"skipped (no transcripts): {path.name}")
            continue
        rows = [json.loads(r) for r in path.read_text(encoding="utf-8").splitlines()]
        s = score([r["ref"] for r in rows], [r["hyp"] for r in rows], test.lang)
        lines[i] = _row(run_date, test, model_name, s, speed, where)
        print(lines[i])
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
