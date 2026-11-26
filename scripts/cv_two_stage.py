"""Cross-validate the two-stage model (quiet/active design) on the fixed session folds.

    python3 scripts/cv_two_stage.py                       # repeat (fold seed) 0
    python3 scripts/cv_two_stage.py --repeats 0 1 2 --name quiet_active
    python3 scripts/cv_two_stage.py --repeats 0 1 2 --name quiet_active_imu --imu --baseline quiet_active
    python3 scripts/cv_two_stage.py --repeats 0 1 2 --name quiet_active_aug60 --aug60 --baseline quiet_active
    python3 scripts/cv_two_stage.py --repeats 0 1 2 --name quiet_active_car --car --baseline quiet_active
    python3 scripts/cv_two_stage.py --repeats 0 1 2 --name quiet_active_td --td --baseline quiet_active
    python3 scripts/cv_two_stage.py --repeats 0 1 2 --name car_qp --car --car-models quiet,pinch --val-aug augch
    python3 scripts/cv_two_stage.py --repeats 0 1 2 --name car_qp_augch --car --car-models quiet,pinch --aug augch --val-aug augch

Needs data/features/*.pkl (scripts/build_features.py). Prints, averaged over the
repeats: stage 1 (quiet vs active), the rest-vs-pinch model on true rest and
pinch windows, the active model on true active windows, and the chained
six-class macro F1 (the competition metric) by validation set, setup (50/60 Hz)
and session. Writes runs/<name>/ with the out-of-fold probabilities and a
metrics.json.

--imu adds the per-axis IMU features (data/features/imu_*.pkl, from
scripts/build_imu_features.py) to stage 1 and the active branch only; the
rest-vs-pinch model is unchanged. --aug60 adds the simulated 60 Hz-like
windows (data/features/aug60_train.pkl, from scripts/build_aug60.py) to the
training data of all three models; in each fold it drops those whose window or
noise comes from a held-out session. --car adds the common-average-reference
features (data/features/car_*.pkl, from scripts/build_car_features.py) to all
three models (--car-models to choose which); --td the time-domain features (data/features/td_*.pkl, from
scripts/build_td_features.py) to all three. --aug NAME adds the windows in
data/features/NAME_train.pkl to the training data of every fold (--aug60 is
--aug aug60), dropping those from held-out sessions. --val-aug NAME also
predicts the corrupted copies of the validation windows in
data/features/NAME_val.pkl (scripts/build_aug_channel.py) with each fold's
models and scores them as a stress test. --baseline NAME prints the change against
runs/NAME/metrics.json, repeat by repeat (same seeds and windows).
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
from aura.io import LABELS  # noqa: E402
from aura.twostage import ACTIVE, QUIET, TwoStage, combine  # noqa: E402

KINDS = ("val", "val_late", "val_steady")
PARTS = ("p_quiet", "p_pinch", "p_active")


def macro_f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def _empty(n):
    return {"p_quiet": np.full(n, np.nan), "p_pinch": np.full(n, np.nan), "p_active": np.full((n, len(ACTIVE)), np.nan)}


def run_repeat(train, val, folds, seed, gain_aug, extra=(), aug=None, extra_for=("quiet", "active"), val_aug=None):
    """Out-of-fold parts for val (and, with val_aug, for its corrupted copy)."""
    parts = _empty(len(val))
    parts_aug = _empty(len(val)) if val_aug is not None else None
    for fo in folds:
        tr = ~train.session.isin(fo)
        va = val.session.isin(fo).to_numpy()
        fit_on = train[tr]
        if aug is not None:
            keep = ~aug.session.isin(fo)
            if "noise_session" in aug.columns:
                keep &= ~aug.noise_session.isin(fo)
            fit_on = pd.concat([fit_on, aug.loc[keep, train.columns]], ignore_index=True)
        m = TwoStage(seed=seed, gain_aug=gain_aug, extra=extra, extra_for=extra_for).fit(fit_on)
        p = m.predict_parts(val[va])
        for k in PARTS:
            parts[k][va] = p[k]
        if val_aug is not None:
            p = m.predict_parts(val_aug[va])
            for k in PARTS:
                parts_aug[k][va] = p[k]
    return parts, parts_aug


def score(val, parts, setup):
    out = {}
    y = val.label.to_numpy()
    sixty = val.session.map(setup).to_numpy() == 60
    pred_all = combine(parts)
    for kind in KINDS:
        k = (val.kind == kind).to_numpy()
        yq = np.isin(y[k], QUIET).astype(int)
        r = {"stage1_BA": balanced_accuracy_score(yq, (parts["p_quiet"][k] >= 0.5).astype(int)),
             "stage1_AUC": roc_auc_score(yq, parts["p_quiet"][k])}
        q = k & np.isin(y, QUIET)
        r["rest_vs_pinch_AUC"] = roc_auc_score(y[q] == "pinch", parts["p_pinch"][q])
        r["rest_vs_pinch_BA"] = balanced_accuracy_score(y[q] == "pinch", parts["p_pinch"][q] >= 0.5)
        a = k & np.isin(y, ACTIVE)
        r["active_F1"] = f1_score(y[a], np.array(ACTIVE)[parts["p_active"][a].argmax(1)], labels=ACTIVE, average="macro")
        pred = pred_all[k]
        r["F1"] = macro_f1(y[k], pred)
        r["F1_50Hz"] = macro_f1(y[k][~sixty[k]], pred[~sixty[k]])
        r["F1_60Hz"] = macro_f1(y[k][sixty[k]], pred[sixty[k]])
        ss = val.session.to_numpy()[k]
        r["F1_session_median"] = float(np.median([macro_f1(y[k][ss == s], pred[ss == s]) for s in np.unique(ss)]))
        r["pinch_F1"] = f1_score(y[k], pred, labels=["pinch"], average=None, zero_division=0)[0]
        out[kind] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, nargs="+", default=[0])
    ap.add_argument("--no-gain-aug", action="store_true", help="train without the random-gain augmentation")
    ap.add_argument("--name", default="quiet_active")
    ap.add_argument("--imu", action="store_true", help="add the per-axis IMU features to stage 1 and the active branch")
    ap.add_argument("--aug60", action="store_true", help="add the simulated 60 Hz-like training windows")
    ap.add_argument("--car", action="store_true", help="add the common-average-reference features to all three models")
    ap.add_argument("--td", action="store_true", help="add the time-domain features to all three models")
    ap.add_argument("--car-models", default="quiet,pinch,active",
                    help="with --car: the models that get the C features (quiet = stage 1, pinch, active)")
    ap.add_argument("--aug", help="add data/features/<NAME>_train.pkl to the training data (e.g. augch)")
    ap.add_argument("--val-aug", help="also score the corrupted validation windows in data/features/<NAME>_val.pkl")
    ap.add_argument("--baseline", help="runs/<name> to compare with, repeat by repeat")
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_pickle(ROOT / "data/features/train.pkl")
    val = pd.read_pickle(ROOT / "data/features/val.pkl")
    extra, extra_for = (), ("quiet", "active")
    if args.imu + args.car + args.td > 1:
        sys.exit("use one of --imu, --car and --td")
    for flag, file, prefix, models, script, what in (
            (args.imu, "imu", "I_", ("quiet", "active"), "build_imu_features.py",
             "per-axis IMU features to stage 1 and the active branch"),
            (args.car, "car", "C_", tuple(args.car_models.split(",")), "build_car_features.py",
             f"common-average-reference features to {args.car_models}"),
            (args.td, "td", "T_", ("quiet", "pinch", "active"), "build_td_features.py",
             "time-domain features to all three models")):
        if not flag:
            continue
        meta = ["session", "seg", "label", "kind", "offset", "n_rows"]
        add = {}
        for name, df in (("train", train), ("val", val)):
            add[name] = pd.read_pickle(ROOT / f"data/features/{file}_{name}.pkl")
            if len(add[name]) != len(df) or not (add[name][meta].to_numpy() == df[meta].to_numpy()).all():
                sys.exit(f"{file}_{name}.pkl does not match {name}.pkl; rerun scripts/{script}")
        cols = [c for c in add["train"].columns if c.startswith(prefix)]
        train = pd.concat([train, add["train"][cols]], axis=1)
        val = pd.concat([val, add["val"][cols]], axis=1)
        extra, extra_for = (prefix,), models
        print(f"added {len(cols)} {what}", flush=True)
    if set(tuple(args.car_models.split(","))) - {"quiet", "pinch", "active"}:
        sys.exit("--car-models takes a comma list of quiet, pinch, active")
    if args.aug60:
        if args.aug and args.aug != "aug60":
            sys.exit("use one of --aug60 and --aug")
        args.aug = "aug60"
    aug = val_aug = None
    if args.aug:
        aug = pd.read_pickle(ROOT / f"data/features/{args.aug}_train.pkl")
        missing = [c for c in train.columns if c not in aug.columns]
        if missing:
            sys.exit(f"{args.aug}_train.pkl lacks {len(missing)} of the training columns (e.g. {missing[0]}); rebuild it")
        print(f"extra training windows from {args.aug}_train.pkl: {len(aug)} (dropped per fold when held out)", flush=True)
    if args.val_aug:
        val_aug = pd.read_pickle(ROOT / f"data/features/{args.val_aug}_val.pkl")
        meta = ["session", "seg", "label", "kind", "offset", "n_rows"]
        if len(val_aug) != len(val) or not (val_aug[meta].to_numpy() == val[meta].to_numpy()).all():
            sys.exit(f"{args.val_aug}_val.pkl does not match val.pkl's windows; rebuild it")
        missing = [c for c in val.columns if c not in val_aug.columns]
        if missing:
            sys.exit(f"{args.val_aug}_val.pkl lacks {len(missing)} of the validation columns (e.g. {missing[0]})")
        val_aug = val_aug[list(val.columns)]
        print(f"stress test: {len(val_aug)} corrupted validation windows from {args.val_aug}_val.pkl", flush=True)
    cfg = read_folds(ROOT / "config/folds.json")
    out_dir = ROOT / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    allm, allm_aug = {}, {}
    for rep in args.repeats:
        parts, parts_aug = run_repeat(train, val, cfg["repeats"][rep], seed=rep, gain_aug=not args.no_gain_aug,
                                      extra=extra, aug=aug, extra_for=extra_for, val_aug=val_aug)
        oof = val[["session", "seg", "label", "kind", "offset"]].copy()
        oof["p_quiet"], oof["p_pinch"] = parts["p_quiet"], parts["p_pinch"]
        for j, g in enumerate(ACTIVE):
            oof[f"p_{g}"] = parts["p_active"][:, j]
        oof.to_pickle(out_dir / f"oof_repeat{rep}.pkl")
        allm[rep] = score(val, parts, cfg["setup_hz"])
        if parts_aug is not None:
            oa = val[["session", "seg", "label", "kind", "offset"]].copy()
            oa["p_quiet"], oa["p_pinch"] = parts_aug["p_quiet"], parts_aug["p_pinch"]
            for j, g in enumerate(ACTIVE):
                oa[f"p_{g}"] = parts_aug["p_active"][:, j]
            oa.to_pickle(out_dir / f"oof_valaug_repeat{rep}.pkl")
            allm_aug[rep] = score(val, parts_aug, cfg["setup_hz"])
        print(f"repeat {rep} done ({time.time() - t0:.0f} s)", flush=True)

    print("\nMean over repeats", args.repeats, "(sessions held out)")
    tab = pd.DataFrame({kind: pd.DataFrame([allm[r][kind] for r in allm]).mean() for kind in KINDS})
    print(tab.round(3).to_string())
    if len(allm) > 1:
        print("\nsix-class F1 per repeat, gestures at 0.7 s:", [round(allm[r]["val"]["F1"], 3) for r in allm])
    if allm_aug:
        print(f"\nSTRESS TEST: the same windows with one bad electrode ({args.val_aug}_val.pkl), mean over repeats")
        tab = pd.DataFrame({kind: pd.DataFrame([allm_aug[r][kind] for r in allm_aug]).mean() for kind in KINDS})
        print(tab.round(3).to_string())
        print("six-class F1 per repeat, gestures at 0.7 s:", [round(allm_aug[r]["val"]["F1"], 3) for r in allm_aug])

    rep = args.repeats[0]
    oof = pd.read_pickle(out_dir / f"oof_repeat{rep}.pkl")
    k = (oof.kind == "val").to_numpy()
    parts = {"p_quiet": oof.p_quiet.to_numpy()[k], "p_pinch": oof.p_pinch.to_numpy()[k],
             "p_active": oof[[f"p_{g}" for g in ACTIVE]].to_numpy()[k]}
    pred, y = combine(parts), oof.label.to_numpy()[k]
    print(f"\nper-class F1, gestures at 0.7 s (repeat {rep}):",
          dict(zip(LABELS, f1_score(y, pred, labels=LABELS, average=None).round(3))))
    print("confusion matrix (rows = true):")
    print(pd.DataFrame(confusion_matrix(y, pred, labels=LABELS), index=LABELS, columns=LABELS).to_string())
    (out_dir / "metrics.json").write_text(json.dumps({"repeats": {str(r): m for r, m in allm.items()},
                                                      "gain_aug": not args.no_gain_aug, "extra": list(extra),
                                                      "aug60": bool(args.aug60), "aug": args.aug, "val_aug": args.val_aug,
                                                      "extra_for": list(extra_for) if extra else [],
                                                      "stress_repeats": {str(r): m for r, m in allm_aug.items()}}, indent=1))
    if args.baseline:
        base = json.loads((ROOT / "runs" / args.baseline / "metrics.json").read_text())["repeats"]
        keys = ["stage1_BA", "stage1_AUC", "active_F1", "F1", "F1_50Hz", "F1_60Hz", "pinch_F1"]
        print(f"\nchange vs runs/{args.baseline}, per repeat (this run minus baseline):")
        for kind in ("val", "val_late"):
            d = pd.DataFrame({k: [allm[r][kind][k] - base[str(r)][kind][k] for r in allm if str(r) in base] for k in keys},
                             index=[f"repeat {r}" for r in allm if str(r) in base])
            d.loc["mean"] = d.mean()
            print(f"gestures at {'0.7' if kind == 'val' else '0.9'} s:")
            print(d.round(3).to_string())
    print(f"\nsaved {out_dir} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
