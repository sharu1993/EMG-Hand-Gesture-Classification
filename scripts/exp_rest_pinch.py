"""Compare variants of the rest-vs-pinch model, keeping the other two models fixed.

    python3 scripts/exp_rest_pinch.py --repeats 0 1 2       # needs runs/quiet_active

Refits only the rest-vs-pinch model per fold; stage 1 (quiet vs active) and the
active-gesture probabilities come from runs/<--trees>/oof_repeat<r>.pkl. Every
variant trains on equal numbers of rest and pinch windows with random-gain
augmentation, as the current model:
  current         one model, 300 trees (aura.twostage)
  capacity        one model, 600 trees, up to 63 leaves
  bagging         5 models, each on all pinch windows and a different fifth of
                  the rest windows (every rest window used once), averaged
  bagging+capacity
Prints the rest-vs-pinch AUC on true rest and pinch windows, pinch F1 and the
chained six-class macro F1, averaged over repeats and per repeat.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.folds import read_folds  # noqa: E402
from aura.io import LABELS  # noqa: E402
from aura.twostage import ACTIVE, QUIET, combine, feature_cols, gain_augment  # noqa: E402

KINDS = ("val", "val_late", "val_steady")


def small(seed):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=seed)


def big(seed):
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                          l2_regularization=1.0, random_state=seed)


VARIANTS = {"current": (small, 1), "capacity": (big, 1), "bagging": (small, 5), "bagging+capacity": (big, 5)}


def rvp_oof(train, val, cols, folds, rep, make, bags):
    lab = train.label.to_numpy()
    X, Xv = train[cols].to_numpy(float), val[cols].to_numpy(float)
    p = np.zeros(len(val))
    rng = np.random.default_rng(rep + 1)
    for fo in folds:
        tr = ~train.session.isin(fo).to_numpy()
        va = val.session.isin(fo).to_numpy()
        ip = np.flatnonzero(tr & (lab == "pinch"))
        rest = rng.permutation(np.flatnonzero(tr & (lab == "rest")))
        chunks = [rest[:len(ip)]] if bags == 1 else np.array_split(rest, bags)
        for b, ch in enumerate(chunks):
            ir = ch if len(ch) <= len(ip) else rng.choice(ch, size=len(ip), replace=False)
            idx = np.r_[ip, ir]
            Xa = gain_augment(X[idx], cols, np.random.default_rng(rep * 100 + b))
            m = make(rep * 10 + b).fit(Xa, (lab[idx] == "pinch").astype(int))
            p[va] += m.predict_proba(Xv[va])[:, 1] / len(chunks)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, nargs="+", default=[0])
    ap.add_argument("--trees", default="quiet_active")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_pickle(ROOT / "data/features/train.pkl")
    val = pd.read_pickle(ROOT / "data/features/val.pkl")
    cols = feature_cols(train)
    cfg = read_folds(ROOT / "config/folds.json")
    y = val.label.to_numpy()
    rows = []
    for rep in args.repeats:
        trees = pd.read_pickle(ROOT / "runs" / args.trees / f"oof_repeat{rep}.pkl")
        if not (trees[["session", "seg", "kind"]].to_numpy() == val[["session", "seg", "kind"]].to_numpy()).all():
            sys.exit(f"runs/{args.trees}/oof_repeat{rep}.pkl does not match the current validation windows")
        for name in args.variants:
            make, bags = VARIANTS[name]
            p = rvp_oof(train, val, cols, cfg["repeats"][rep], rep, make, bags)
            parts = {"p_quiet": trees.p_quiet.to_numpy(), "p_pinch": p,
                     "p_active": trees[[f"p_{g}" for g in ACTIVE]].to_numpy()}
            pred = combine(parts)
            r = {"repeat": rep, "variant": name}
            for kind in KINDS:
                k = (val.kind == kind).to_numpy()
                q = k & np.isin(y, QUIET)
                r[f"AUC {kind}"] = roc_auc_score(y[q] == "pinch", p[q])
                r[f"six-class {kind}"] = f1_score(y[k], pred[k], labels=LABELS, average="macro")
                r[f"pinch F1 {kind}"] = f1_score(y[k], pred[k], labels=["pinch"], average=None)[0]
            rows.append(r)
            print(f"repeat {rep}, {name} done ({time.time() - t0:.0f} s)", flush=True)
    T = pd.DataFrame(rows)
    out = ROOT / "runs" / "exp_rest_pinch"
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "summary.csv", index=False)
    show = ["AUC val", "AUC val_steady", "pinch F1 val", "six-class val", "six-class val_late", "six-class val_steady"]
    print("\nMean over repeats", args.repeats)
    print(T.groupby("variant")[show].mean().reindex(args.variants).round(3).to_string())
    if len(args.repeats) > 1:
        print("\nsix-class F1, gestures at 0.7 s, per repeat:")
        print(T.pivot(index="repeat", columns="variant", values="six-class val")[args.variants].round(3).to_string())
    print(f"\nsaved {out / 'summary.csv'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
