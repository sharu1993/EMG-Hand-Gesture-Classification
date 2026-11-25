"""Session-grouped folds, one 60 Hz session per fold (doc "Cross-validation").

Written once to config/folds.json and reused by every model, so all scores are
on the same splits. Repeats = different seeds for the assignment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_FOLDS = 7
SEEDS = (0, 1, 2)


def make_folds(setup: dict[int, int], n_folds: int = N_FOLDS, seed: int = 0) -> list[list[int]]:
    """Each fold gets an equal share of 60 Hz sessions, then the 50 Hz sessions
    are dealt round robin, both after a seeded shuffle."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    i = 0
    for hz in (60, 50):
        sess = sorted(s for s, v in setup.items() if v == hz)
        for s in rng.permutation(sess):
            folds[i % n_folds].append(int(s))
            i += 1
    return [sorted(f) for f in folds]


def write_folds(path: Path, setup: dict[int, int], seeds=SEEDS, n_folds: int = N_FOLDS) -> dict:
    cfg = {"n_folds": n_folds,
           "setup_hz": {str(s): int(v) for s, v in sorted(setup.items())},
           "repeats": {str(seed): make_folds(setup, n_folds, seed) for seed in seeds}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1))
    return cfg


def read_folds(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg["setup_hz"] = {int(s): v for s, v in cfg["setup_hz"].items()}
    cfg["repeats"] = {int(k): v for k, v in cfg["repeats"].items()}
    return cfg
