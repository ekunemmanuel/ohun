"""The `ohun` command.

    ohun catalog [--kind dataset|model] [--lang yor] [--task stt|tts] [--commercial]
    ohun normalize --lang yor [--level full|no_tone|bare] "Ọkọ̀ náà ti dé"
    ohun score --lang yor references.txt hypotheses.txt   (one sentence per line)
"""

import argparse
import sys
from pathlib import Path

from ohun import catalog, languages, metrics, text


def _catalog(args: argparse.Namespace) -> None:
    entries = catalog.find(
        catalog.load(),
        kind=args.kind,
        lang=args.lang,
        task=args.task,
        commercial_only=args.commercial,
    )
    for e in entries:
        use = "ok to ship" if e.commercial else "research only"
        flags = "  [gated]" if e.gated else ""
        langs = ",".join(e.langs) or "any"
        print(f"{e.kind:7}  {e.id:18} {'/'.join(e.tasks):7} {langs:16} {e.size():22} {e.license:16} {use}{flags}")
        print(f"         {e.role}")
    print(f"\n{len(entries)} entries")


def _normalize(args: argparse.Namespace) -> None:
    print(text.for_scoring(" ".join(args.text), args.lang, args.level))


def _score(args: argparse.Namespace) -> None:
    refs = Path(args.references).read_text(encoding="utf-8").splitlines()
    hyps = Path(args.hypotheses).read_text(encoding="utf-8").splitlines()
    print(metrics.score(refs, hyps, args.lang).summary())


def _baseline(args: argparse.Namespace) -> None:
    from ohun import baseline  # imports torch, so only when asked

    if args.model not in baseline.MODELS:
        raise ValueError(f"Unknown model {args.model!r}. Known: {', '.join(baseline.MODELS)}")
    baseline.run(args.lang, args.model, args.n)


def _rescore(_: argparse.Namespace) -> None:
    from ohun import baseline

    baseline.rescore()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ohun", description="Speech tools for Nigerian languages")
    sub = parser.add_subparsers(dest="command", required=True)
    lang_codes = list(languages.LANGUAGES)

    p = sub.add_parser("catalog", help="list datasets and models with their licenses")
    p.add_argument("--kind", choices=["dataset", "model"])
    p.add_argument("--lang", choices=lang_codes)
    p.add_argument("--task", choices=["stt", "tts"])
    p.add_argument("--commercial", action="store_true", help="only entries we could ship")
    p.set_defaults(run=_catalog)

    p = sub.add_parser("normalize", help="show text the way the scorer sees it")
    p.add_argument("--lang", choices=lang_codes, required=True)
    p.add_argument("--level", choices=["full", "no_tone", "bare"], default="full")
    p.add_argument("text", nargs="+")
    p.set_defaults(run=_normalize)

    p = sub.add_parser("score", help="WER/CER of hypotheses against references")
    p.add_argument("--lang", choices=lang_codes, required=True)
    p.add_argument("references")
    p.add_argument("hypotheses")
    p.set_defaults(run=_score)

    p = sub.add_parser("baseline", help="score an existing model (needs: uv run --group eval)")
    p.add_argument("--lang", choices=["yor", "pcm"], required=True)
    p.add_argument("--model", required=True, help="whisper-small, whisper-turbo, mms-asr, ncair-yoruba-asr")
    p.add_argument("--n", type=int, default=50, help="number of test clips")
    p.set_defaults(run=_baseline)

    p = sub.add_parser("rescore", help="recompute docs/BASELINES.md from saved transcripts")
    p.set_defaults(run=_rescore)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except ValueError as err:
        sys.exit(f"ohun: {err}")


if __name__ == "__main__":
    main()
