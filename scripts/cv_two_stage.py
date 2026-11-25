"""Cross-validate the two-stage model on the fixed session folds.

    python scripts/cv_two_stage.py                 # repeat (fold seed) 0
    python scripts/cv_two_stage.py --repeats 0 1 2

Needs data/features/*.pkl (scripts/build_features.py). Prints, per repeat and
averaged: stage 1 on its own, stage 2 on its own (true gesture windows), and the
chained six-class macro F1 (the competition metric) for both combining rules,
by validation set, setup (50/60 Hz) and session. Writes runs/<name>/ with the
out-of-fold probabilities and a metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.folds import read_folds  # noqa: E402
from aura.io import GESTURES, LABELS  # noqa: E402
from aura.twostage import TwoStage, combine  # noqa: E402

KINDS = ("val", "val_late", "val_steady")


def macro_f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def run_repeat(train, val, folds, seed, gain_aug):
    p1 = np.full(len(val), np.nan)
    p2 = np.full((len(val), len(GESTURES)), np.nan)
    for fo in folds:
        tr = ~train.session.isin(fo)
        va = val.session.isin(fo).to_numpy()
        m = TwoStage(seed=seed, gain_aug=gain_aug).fit(train[tr])
        p1[va], p2[va] = m.predict_parts(val[va])
    return p1, p2


def score(val, p1, p2, setup):
    out = {}
    y = val.label.to_numpy()
    sixty = val.session.map(setup).to_numpy() == 60
    for kind in KINDS:
        k = (val.kind == kind).to_numpy()
        yg = (y[k] != "rest").astype(int)
        r = {"stage1_BA": balanced_accuracy_score(yg, (p1[k] >= 0.5).astype(int)),
             "stage1_AUC": roc_auc_score(yg, p1[k])}
        g = k & (y != "rest")
        r["stage2_F1"] = f1_score(y[g], np.array(GESTURES)[p2[g].argmax(1)], labels=GESTURES, average="macro")
        for rule in ("gate", "soft"):
            pred = combine(p1[k], p2[k], rule)
            r[f"F1_{rule}"] = macro_f1(y[k], pred)
            r[f"F1_{rule}_50Hz"] = macro_f1(y[k][~sixty[k]], pred[~sixty[k]])
            r[f"F1_{rule}_60Hz"] = macro_f1(y[k][sixty[k]], pred[sixty[k]])
            ss = val.session.to_numpy()[k]
            r[f"F1_{rule}_session_median"] = float(np.median([macro_f1(y[k][ss == s], pred[ss == s]) for s in np.unique(ss)]))
        out[kind] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, nargs="+", default=[0])
    ap.add_argument("--no-gain-aug", action="store_true", help="stage 1 without the random-gain augmentation")
    ap.add_argument("--name", default="two_stage_baseline")
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_pickle(ROOT / "data/features/train.pkl")
    val = pd.read_pickle(ROOT / "data/features/val.pkl")
    cfg = read_folds(ROOT / "config/folds.json")
    out_dir = ROOT / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    allm = {}
    for rep in args.repeats:
        p1, p2 = run_repeat(train, val, cfg["repeats"][rep], seed=rep, gain_aug=not args.no_gain_aug)
        oof = val[["session", "seg", "label", "kind", "offset"]].copy()
        oof["p_gesture"] = p1
        for j, gname in enumerate(GESTURES):
            oof[f"p_{gname}"] = p2[:, j]
        oof.to_pickle(out_dir / f"oof_repeat{rep}.pkl")
        allm[rep] = score(val, p1, p2, cfg["setup_hz"])
        print(f"repeat {rep} done ({time.time() - t0:.0f} s)", flush=True)

    print("\nMean over repeats", args.repeats, "(sessions held out)")
    tab = pd.DataFrame({kind: pd.DataFrame([allm[r][kind] for r in allm]).mean() for kind in KINDS})
    print(tab.round(3).to_string())
    if len(allm) > 1:
        print("\nend-to-end test-like F1 per repeat (gate / soft):",
              [(round(allm[r]["val"]["F1_gate"], 3), round(allm[r]["val"]["F1_soft"], 3)) for r in allm])

    # confusion matrix and per-class F1, test-like windows, gate rule, first repeat
    rep = args.repeats[0]
    oof = pd.read_pickle(out_dir / f"oof_repeat{rep}.pkl")
    k = (oof.kind == "val").to_numpy()
    pred = combine(oof.p_gesture.to_numpy()[k], oof[[f"p_{g}" for g in GESTURES]].to_numpy()[k], "gate")
    y = oof.label.to_numpy()[k]
    print(f"\nper-class F1, test-like, gate (repeat {rep}):",
          dict(zip(LABELS, f1_score(y, pred, labels=LABELS, average=None).round(3))))
    print("confusion matrix (rows = true):")
    print(pd.DataFrame(confusion_matrix(y, pred, labels=LABELS), index=LABELS, columns=LABELS).to_string())
    (out_dir / "metrics.json").write_text(json.dumps({"repeats": {str(r): m for r, m in allm.items()},
                                                      "gain_aug": not args.no_gain_aug}, indent=1))
    print(f"\nsaved {out_dir} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
