"""Cross-validate a TCN for one of the three decisions of the quiet/active model
on the fixed session folds, and score it in the chain against the trees.

    python3 scripts/cv_sequence.py --repeats 0 1 2 --frames relations                    # active gestures
    python3 scripts/cv_sequence.py --repeats 0 1 2 --frames relations --task rest_pinch
    python3 scripts/cv_sequence.py --repeats 0 1 2 --frames relations --task quiet
    python3 scripts/cv_sequence.py --repeats 0 1 2 --frames relations_car --task rest_pinch --seeds 3

Tasks (each trained the same way as the matching tree model in aura/twostage.py):
  active      fist / open / point / thumb-up, on active windows            -> p_active
  rest_pinch  rest vs pinch, on all pinch windows and an equal random subset
              of rest windows (a different subset per seed)                -> p_pinch
  quiet       quiet (rest, pinch) vs active, on all windows                -> p_quiet

Needs the frames from scripts/build_frames.py (--kind to match --frames) and
the tree model's out-of-fold probabilities in runs/<--trees>/oof_repeat<r>.pkl
for the same repeats. Trains on the Apple GPU (MPS) when available.

For the chosen decision, prints the trees alone, the TCN alone and the average
of the two (the other two decisions stay the trees'): the decision's own score
and the chained six-class macro F1, averaged over repeats and per repeat.
Writes runs/<name>/seq_oof_repeat<r>.pkl and summary.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.folds import read_folds  # noqa: E402
from aura.io import LABELS  # noqa: E402
from aura.sequence import pick_device, predict_tcn, train_tcn  # noqa: E402
from aura.twostage import ACTIVE, QUIET, combine  # noqa: E402

KINDS = ("val", "val_late", "val_steady")
PART = {"active": "p_active", "rest_pinch": "p_pinch", "quiet": "p_quiet"}


def train_rows(task, labels, train_mask, rng):
    """Indices of the training windows and their class ids for one fold."""
    if task == "active":
        idx = np.flatnonzero(train_mask & np.isin(labels, ACTIVE))
        return idx, np.array([ACTIVE.index(l) for l in labels[idx]])
    if task == "rest_pinch":
        ip = np.flatnonzero(train_mask & (labels == "pinch"))
        ir = rng.choice(np.flatnonzero(train_mask & (labels == "rest")), size=len(ip), replace=False)
        idx = np.r_[ip, ir]
        return idx, (labels[idx] == "pinch").astype(int)
    idx = np.flatnonzero(train_mask)
    return idx, np.isin(labels[idx], QUIET).astype(int)


def decision_score(task, y, p, k):
    """The decision's own score on the validation windows selected by k."""
    if task == "active":
        a = k & np.isin(y, ACTIVE)
        return {"active F1": f1_score(y[a], np.array(ACTIVE)[p[a].argmax(1)], labels=ACTIVE, average="macro")}
    if task == "rest_pinch":
        q = k & np.isin(y, QUIET)
        return {"AUC": roc_auc_score(y[q] == "pinch", p[q]),
                "BA": balanced_accuracy_score(y[q] == "pinch", p[q] >= 0.5)}
    yq = np.isin(y[k], QUIET)
    return {"AUC": roc_auc_score(yq, p[k]), "BA": balanced_accuracy_score(yq, p[k] >= 0.5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(PART), default="active")
    ap.add_argument("--repeats", type=int, nargs="+", default=[0])
    ap.add_argument("--trees", default="quiet_active", help="runs/<name> holding the tree model's out-of-fold files")
    ap.add_argument("--frames", choices=["power", "relations", "relations_car"], default="power")
    ap.add_argument("--name", default=None, help="default tcn_<task>_<frames> (tcn_active for active/power)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1, help="TCNs per fold, probabilities averaged")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    device = pick_device(args.device)
    feat = ROOT / "data/features"
    tr_meta = pd.read_pickle(feat / "train.pkl")[["session", "label", "kind"]]
    va_meta = pd.read_pickle(feat / "val.pkl")[["session", "seg", "label", "kind"]]
    prefix = "frames_" if args.frames == "power" else f"frames_{args.frames}_"
    if args.name is None:
        args.name = "tcn_active" if (args.task, args.frames) == ("active", "power") else \
            (f"tcn_active_{args.frames}" if args.task == "active" else f"tcn_{args.task}_{args.frames}")
    ftr, fva = np.load(feat / f"{prefix}train.npz"), np.load(feat / f"{prefix}val.npz")
    Xtr, Mtr, Xva, Mva = ftr["X"], ftr["M"], fva["X"], fva["M"]
    cfg = read_folds(ROOT / "config/folds.json")
    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    labels = tr_meta.label.to_numpy()
    y = va_meta.label.to_numpy()
    part = PART[args.task]
    print(f"device {device}; task {args.task}; {args.frames} frames ({Xtr.shape[1]} x {Xtr.shape[2]}); "
          f"{args.seeds} seed(s); {len(va_meta)} validation windows", flush=True)

    rows = []
    for rep in args.repeats:
        trees = pd.read_pickle(ROOT / "runs" / args.trees / f"oof_repeat{rep}.pkl")
        if not (trees[["session", "seg", "kind"]].to_numpy() == va_meta[["session", "seg", "kind"]].to_numpy()).all():
            sys.exit(f"runs/{args.trees}/oof_repeat{rep}.pkl does not match the current validation windows")
        tree_parts = {"p_quiet": trees.p_quiet.to_numpy(), "p_pinch": trees.p_pinch.to_numpy(),
                      "p_active": trees[[f"p_{g}" for g in ACTIVE]].to_numpy()}
        p_seq = np.zeros_like(tree_parts[part], dtype=float)
        for k, fo in enumerate(cfg["repeats"][rep]):
            trm = ~tr_meta.session.isin(fo).to_numpy()
            va = va_meta.session.isin(fo).to_numpy()
            for j in range(args.seeds):
                seed = rep * 100 + k * 10 + j
                idx, yc = train_rows(args.task, labels, trm, np.random.default_rng(seed))
                model, std = train_tcn(Xtr[idx], Mtr[idx], yc, device, seed=seed, epochs=args.epochs,
                                       verbose=args.verbose)
                pr = predict_tcn(model, std, Xva[va], Mva[va], device)
                p_seq[va] += (pr if args.task == "active" else pr[:, 1]) / args.seeds
            print(f"repeat {rep} fold {k + 1}/{len(cfg['repeats'][rep])} ({time.time() - t0:.0f} s)", flush=True)
        oof = va_meta.copy()
        if args.task == "active":
            for j, g in enumerate(ACTIVE):
                oof[f"p_{g}"] = p_seq[:, j]
        else:
            oof[part] = p_seq
        oof.to_pickle(out / f"seq_oof_repeat{rep}.pkl")
        for name, p in (("trees", tree_parts[part]), ("TCN", p_seq), ("average", (tree_parts[part] + p_seq) / 2)):
            parts = dict(tree_parts, **{part: p})
            pred = combine(parts)
            r = {"repeat": rep, "model": name}
            for kind in KINDS:
                kk = (va_meta.kind == kind).to_numpy()
                for m, v in decision_score(args.task, y, p, kk).items():
                    r[f"{m} {kind}"] = v
                r[f"six-class {kind}"] = f1_score(y[kk], pred[kk], labels=LABELS, average="macro")
                if args.task != "active":
                    r[f"pinch F1 {kind}"] = f1_score(y[kk], pred[kk], labels=["pinch"], average=None)[0]
            rows.append(r)

    T = pd.DataFrame(rows)
    T.to_csv(out / "summary.csv", index=False)
    cols = [c for c in T.columns if c not in ("repeat", "model")]
    print(f"\nMean over repeats {args.repeats} (task {args.task}; the other decisions are the trees')")
    print(T.groupby("model")[cols].mean().reindex(["trees", "TCN", "average"]).round(3).T.to_string())
    if len(args.repeats) > 1:
        print("\nsix-class F1, gestures at 0.7 s, per repeat:")
        print(T.pivot(index="repeat", columns="model", values="six-class val")[["trees", "TCN", "average"]].round(3).to_string())
    print(f"\nsaved {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
