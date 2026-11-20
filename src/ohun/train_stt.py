"""Phase 2: fine-tune Whisper on Yoruba + Nigerian Pidgin together. Needs a GPU (Kaggle T4 x2).

Run through kaggle/train-stt/run.py. Data, all commercially usable:

  Pidgin  WaZoBiaSpeech train (streamed shard by shard, never all on disk) + nigerian-pidgin-1.0 train
  Yoruba  FLEURS yo_ng train (toned references only) + BibleTTS Yoruba train (one speaker, toned)

Each training example picks Yoruba with probability `yor_share`, else Pidgin. Training
stops after `max_steps`; then the model is scored on the same test sets as the baselines.
"""

import random
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ohun import languages
from ohun.baseline import SAMPLE_RATE, decode_audio
from ohun.text import has_tone_marks, normalize_yoruba

WAZOBIA = "Africanvoice/naija_african_voices"
PIDGIN_1 = "asr-nigerian-pidgin/nigerian-pidgin-1.0"
BIBLETTS_YOR = "https://www.openslr.org/resources/129/yoruba.tgz"  # 6.4 GB
MAX_SECONDS = 30  # Whisper's window
MAX_LABEL_TOKENS = 440


@dataclass
class Config:
    base: str = "openai/whisper-small"
    out: str = "/kaggle/working/ohun-stt"
    work: str = "/tmp/ohun"  # scratch space, not kept
    max_steps: int = 4000
    batch_per_gpu: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 400
    yor_share: float = 0.5
    pidgin1_share: float = 0.2  # of Pidgin examples, the rest are WaZoBiaSpeech
    workers: int = 4
    seed: int = 0
    # Smoke test on the Mac: tiny model, 20 cached test clips per language, no BibleTTS or
    # WaZoBiaSpeech download. Only checks that the pipeline runs end to end.
    smoke: bool = False

    @classmethod
    def smoke_test(cls) -> "Config":
        return cls(
            base="openai/whisper-tiny", out="runs/smoke/model", work="runs/smoke/work", max_steps=2,
            batch_per_gpu=2, warmup_steps=0, pidgin1_share=1.0, workers=0, smoke=True,
        )


# ------------------------------------------------------------------ data


def _prepare_yoruba(cfg: Config) -> list[tuple[str, str]]:
    """Write every usable Yoruba training clip to disk once; return (audio path, text) pairs."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    work = Path(cfg.work)
    pairs: list[tuple[str, str]] = []

    fleurs_dir = work / "fleurs-yor"
    fleurs_dir.mkdir(parents=True, exist_ok=True)
    split = "test" if cfg.smoke else "train"
    path = hf_hub_download("google/fleurs", f"parquet-data/yo_ng/{split}-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(path, columns=["id", "raw_transcription", "audio"])
    if cfg.smoke:
        table = table.slice(0, 20)
    skipped = 0
    for batch in table.to_batches(64):
        for row in batch.to_pylist():
            text = row["raw_transcription"] or ""
            if not has_tone_marks(text):  # untoned text would teach the model to drop tones
                skipped += 1
                continue
            wav = fleurs_dir / f"{row['id']}-{len(pairs)}.audio"
            wav.write_bytes(row["audio"]["bytes"])
            pairs.append((str(wav), normalize_yoruba(text)))
    print(f"FLEURS Yoruba {split}: {len(pairs)} toned clips, {skipped} untoned skipped", flush=True)
    if cfg.smoke:
        return pairs

    bible_dir = work / "bibletts"
    if not (bible_dir / "yoruba" / "train").exists():
        bible_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading BibleTTS Yoruba (6.4 GB) ...", flush=True)
        with urllib.request.urlopen(BIBLETTS_YOR) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if "/train/" in member.name and member.isfile():
                    tar.extract(member, bible_dir, filter="data")
    n_before = len(pairs)
    for txt in sorted((bible_dir / "yoruba" / "train").rglob("*.txt")):
        flac = txt.with_suffix(".flac")
        text = txt.read_text(encoding="utf-8").strip()
        if flac.exists() and text:
            pairs.append((str(flac), normalize_yoruba(text)))
    print(f"BibleTTS Yoruba train: {len(pairs) - n_before} clips", flush=True)
    return pairs


def _prepare_pidgin1(cfg: Config) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(cfg.work) / "pidgin-1.0"
    out.mkdir(parents=True, exist_ok=True)
    pairs = []
    prefix = "data/test-" if cfg.smoke else "data/train-"
    files = [f for f in HfApi().list_repo_files(PIDGIN_1, repo_type="dataset") if f.startswith(prefix)]
    for f in files:
        table = pq.read_table(hf_hub_download(PIDGIN_1, f, repo_type="dataset"), columns=["sentence", "audio"])
        if cfg.smoke:
            table = table.slice(0, 20)
        for row in table.to_pylist():
            if (row["sentence"] or "").strip():
                wav = out / f"{len(pairs)}.audio"
                wav.write_bytes(row["audio"]["bytes"])
                pairs.append((str(wav), row["sentence"].strip()))
    print(f"nigerian-pidgin-1.0 train: {len(pairs)} clips", flush=True)
    return pairs


def _wazobia_shards() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(WAZOBIA, repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/train-"))


def _make_dataset(cfg: Config, processor, yor: list, pidgin1: list, shards: list[str]):
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download

    tokenizer = processor.tokenizer
    extractor = processor.feature_extractor
    shard_dir = Path(cfg.work) / "wazobia"

    def encode(audio, text: str, lang: str):
        if len(audio) > MAX_SECONDS * SAMPLE_RATE or len(audio) < SAMPLE_RATE // 2:
            return None
        tokenizer.set_prefix_tokens(language=languages.get(lang).whisper, task="transcribe")
        labels = tokenizer(text).input_ids
        if len(labels) > MAX_LABEL_TOKENS:
            return None
        features = extractor(audio, sampling_rate=SAMPLE_RATE).input_features[0]
        return {"input_features": features, "labels": labels}

    def wazobia_rows(rng: random.Random, my_shards: list[str]):
        """Endless: download a shard, yield its rows shuffled, delete it, next shard."""
        while True:
            order = my_shards[:]
            rng.shuffle(order)
            for name in order:
                path = hf_hub_download(WAZOBIA, name, repo_type="dataset", local_dir=shard_dir)
                table = pq.read_table(path, columns=["transcript", "audio_path"])
                Path(path).unlink(missing_ok=True)
                idx = list(range(table.num_rows))
                rng.shuffle(idx)
                for i in idx:
                    row = table.slice(i, 1).to_pylist()[0]
                    if (row["transcript"] or "").strip():
                        yield row["audio_path"]["bytes"], row["transcript"].strip()

    class Mixed(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            wid, nw = (info.id, info.num_workers) if info else (0, 1)
            rng = random.Random(cfg.seed * 1000 + wid)
            wzb = wazobia_rows(rng, shards[wid::nw])
            while True:
                if rng.random() < cfg.yor_share:
                    path, text = rng.choice(yor)
                    lang, data = "yor", Path(path).read_bytes()
                elif rng.random() < cfg.pidgin1_share:
                    path, text = rng.choice(pidgin1)
                    lang, data = "pcm", Path(path).read_bytes()
                else:
                    (data, text), lang = next(wzb), "pcm"
                try:
                    example = encode(decode_audio(data), text, lang)
                except Exception as err:  # one bad file must not kill a 3-hour run
                    print(f"skipped a clip: {err}", flush=True)
                    continue
                if example:
                    yield example

    return Mixed()


class Collator:
    def __init__(self, processor, decoder_start_token_id: int):
        self.processor = processor
        self.start = decoder_start_token_id

    def __call__(self, features: list[dict]):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.start).all():  # the model adds the start token itself
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


# ------------------------------------------------------------------ train + evaluate


def train(cfg: Config) -> Path:
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    yor = _prepare_yoruba(cfg)
    pidgin1 = _prepare_pidgin1(cfg)
    shards = _wazobia_shards()
    print(f"WaZoBiaSpeech: {len(shards)} train shards to stream", flush=True)

    processor = WhisperProcessor.from_pretrained(cfg.base)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.base)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    args = Seq2SeqTrainingArguments(
        output_dir=str(Path(cfg.work) / "checkpoints"),
        per_device_train_batch_size=cfg.batch_per_gpu,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50,
        save_steps=1000,
        save_total_limit=1,
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=cfg.workers,
        remove_unused_columns=False,
        seed=cfg.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=_make_dataset(cfg, processor, yor, pidgin1, shards),
        data_collator=Collator(processor, model.config.decoder_start_token_id),
    )
    trainer.train()

    out = Path(cfg.out)
    model.config.use_cache = True
    trainer.save_model(str(out))  # safetensors
    processor.save_pretrained(str(out))
    shutil.rmtree(Path(cfg.work) / "checkpoints", ignore_errors=True)
    print(f"Saved model to {out}", flush=True)
    return out


def evaluate_trained(model_dir: Path, name: str) -> None:
    from ohun.baseline import ModelSpec, evaluate

    spec = ModelSpec(str(model_dir), "whisper")
    for test_key in ("fleurs-yor", "pidgin-1.0", "wazobia-dev-test"):
        evaluate(test_key, name, spec)
