"""Reads data/catalog.toml: the datasets and models Ohùn knows about, with licenses."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "catalog.toml"


@dataclass
class Entry:
    kind: str  # "dataset" or "model"
    id: str
    source: str
    tasks: list[str]
    license: str
    commercial: bool
    role: str
    name: str = ""
    langs: list[str] = field(default_factory=list)
    hours: dict[str, float] = field(default_factory=dict)
    utterances: int | None = None
    speakers: int | None = None
    gated: bool = False

    def size(self) -> str:
        if self.hours:
            return ", ".join(f"{lang} {h:g}h" for lang, h in self.hours.items())
        if self.utterances:
            return f"{self.utterances:,} clips"
        return ""


def load(path: Path = CATALOG_PATH) -> list[Entry]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return [Entry(kind=kind, **item) for kind in ("dataset", "model") for item in raw.get(kind, [])]


def find(
    entries: list[Entry],
    *,
    kind: str | None = None,
    lang: str | None = None,
    task: str | None = None,
    commercial_only: bool = False,
) -> list[Entry]:
    def keep(e: Entry) -> bool:
        if kind and e.kind != kind:
            return False
        # Models list no languages: they are bases that can learn any of them.
        if lang and e.langs and lang not in e.langs:
            return False
        if task and task not in e.tasks:
            return False
        return e.commercial or not commercial_only

    return [e for e in entries if keep(e)]
