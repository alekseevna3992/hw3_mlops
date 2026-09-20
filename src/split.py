"""Стадия split: разбиение на train/val/test по группам (не по строкам)."""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

from src.config import load_params
from src.contamination import report
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group


def group_split(
    groups: dict[str, list[Example]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[Example]]:
    """Раздать группы целиком в один из сплитов, не разрывая группу.

    Сплит по строкам даёт утечку: одна и та же формулировка встречается
    в разных темах, и после случайного разбиения часть дублей уезжает в test.
    Разбиение по группам гарантирует: если внутри группы остались
    near-дубли, они остаются в одном сплите и не контаминируют test.
    """
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    names = list(ratios)
    assign: dict[str, str] = {}
    start = 0
    for i, name in enumerate(names):
        stop = len(keys) if i == len(names) - 1 else start + round(len(keys) * ratios[name])
        for key in keys[start:stop]:
            assign[key] = name
        start = stop

    buckets: dict[str, list[Example]] = {name: [] for name in names}
    for key, rows in groups.items():
        buckets[assign[key]].extend(rows)
    return buckets


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    groups: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        groups[normalize_group(ex.topic)].append(ex)

    buckets = group_split(groups, cfg["ratios"], cfg["seed"])

    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(dump(ex) + "\n")

    nd = params["clean"]["near_dup"]
    rep = report(
        buckets["train"],
        buckets["test"],
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )

    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(groups),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(ex.topic) for ex in rows}) for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": rep,
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "split: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items())
        + f" (групп {len(groups)}, {metrics['seconds']} с)"
    )


if __name__ == "__main__":
    main()