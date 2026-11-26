"""Cross-validate a linear model for the active branch and its effect on the full model.

    python3 scripts/cv_linear_active.py --repeats 0 1 2 --name linear_active

Model: standardised features, multinomial logistic regression (L2, --C), fit on
the active training windows of each fold. It is linear, so its errors should
differ from the trees' and the TCNs', which is what averaging needs.

Feature sets, both tried in one run:
  S    the scale-free gesture pattern: covariance shape and scale and correlations
       in tangent space per band, plus the IMU summaries (S_*, 202)
  S+C  S plus the common-average-reference covariance shape and scale and the
       common share per band (C_shape, C_scale, C_common; 90). The absolute
       C_logp powers are left out.

Full model scored, as submitted in runs/car_tcn_rp: stage 1 and rest vs pinch
from runs/quiet_active_car, the active trees from runs/quiet_active, averaged
with the active TCNs (runs/tcn_active_relations_3seeds), and p(pinch) averaged
with the rest-vs-pinch TCNs (runs/tcn_rest_pinch_relations). The linear model's
active probabilities are added as a third, equal member of the active average.

Needs data/features/{train,val}.pkl, data/features/car_{train,val}.pkl and those
runs' out-of-fold files. Writes runs/<name>/oof_linear_<set>_repeat<r>.pkl and
summary.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.folds import read_folds  # noqa: E402
from aura.io import LABELS  # noqa: E402
from aura.twostage import ACTIVE, combine  # noqa: E402

META = ["session", "seg", "label", "kind", "offset", "n_rows"]
KEY = ["session", "seg", "kind"]


def feature_sets(cols: list[str]) -> dict[str, list[str]]:
    s = [c for c in cols if c.startswith("S_")]
    c = [c for c in cols if c.startswith(("C_shape", "C_scale", "C_common"))]
    return {"S": s, "S+C": s + c}


def load_oof(run: str, rep: int, prefix: str = "oof_repeat") -> pd.DataFrame:
    return pd.read_pickle(ROOT / "runs" / run / f"{prefix}{rep}.pkl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--C", type=float, default=0.1, help="inverse L2 strength (my pick; smaller = stronger)")
    ap.add_argument("--name", default="linear_active")
    args = ap.parse_args()

    t0 = time.time()
    feat = ROOT / "data/features"
    train, val = pd.read_pickle(feat / "train.pkl"), pd.read_pickle(feat / "val.pkl")
    for name, df in (("train", train), ("val", val)):
        car = pd.read_pickle(feat / f"car_{name}.pkl")
        if len(car) != len(df) or not (car[META].to_numpy() == df[META].to_numpy()).all():
            sys.exit(f"car_{name}.pkl does not match {name}.pkl; rerun scripts/build_car_features.py")
    train = pd.concat([train, pd.read_pickle(feat / "car_train.pkl").filter(like="C_")], axis=1)
    val = pd.concat([val, pd.read_pickle(feat / "car_val.pkl").filter(like="C_")], axis=1)
    sets = feature_sets(list(train.columns))
    print({k: len(v) for k, v in sets.items()}, "features", flush=True)
    cfg = read_folds(ROOT / "config/folds.json")
    sixty = val.session.map(cfg["setup_hz"]).to_numpy() == 60
    out_dir = ROOT / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    y_tr, y = train.label.to_numpy(), val.label.to_numpy()
    rows = []
    for rep in args.repeats:
        car_t, base_t = load_oof("quiet_active_car", rep), load_oof("quiet_active", rep)
        tcn_a = load_oof("tcn_active_relations_3seeds", rep, "seq_oof_repeat")
        tcn_r = load_oof("tcn_rest_pinch_relations", rep, "seq_oof_repeat")
        for d in (car_t, base_t, tcn_a, tcn_r):
            if not (d[KEY].to_numpy() == val[KEY].to_numpy()).all():
                sys.exit("an out-of-fold file does not match val.pkl's windows")
        pa_trees = base_t[[f"p_{g}" for g in ACTIVE]].to_numpy()
        pa_tcn = tcn_a[[f"p_{g}" for g in ACTIVE]].to_numpy()
        q = car_t.p_quiet.to_numpy()
        pp = (car_t.p_pinch.to_numpy() + tcn_r.p_pinch.to_numpy()) / 2

        lin = {}
        for sname, cols in sets.items():
            p = np.full((len(val), len(ACTIVE)), np.nan)
            for fo in cfg["repeats"][rep]:
                tr = (~train.session.isin(fo)).to_numpy() & np.isin(y_tr, ACTIVE)
                va = val.session.isin(fo).to_numpy()
                m = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=3000))
                m.fit(train.loc[tr, cols].to_numpy(float), y_tr[tr])
                order = [list(m.classes_).index(g) for g in ACTIVE]
                p[va] = m.predict_proba(val.loc[va, cols].to_numpy(float))[:, order]
            lin[sname] = p
            o = val[["session", "seg", "label", "kind", "offset"]].copy()
            for j, g in enumerate(ACTIVE):
                o[f"p_{g}"] = p[:, j]
            o.to_pickle(out_dir / f"oof_linear_{sname.replace('+', '')}_repeat{rep}.pkl")
            print(f"repeat {rep}, {sname}: done ({time.time() - t0:.0f} s)", flush=True)

        variants = {"current (trees+TCN)": (pa_trees + pa_tcn) / 2}
        for sname, p in lin.items():
            variants[f"linear {sname} alone"] = p
            variants[f"trees+TCN+linear {sname}"] = (pa_trees + pa_tcn + p) / 3
        for vname, pa in variants.items():
            pred = combine({"p_quiet": q, "p_pinch": pp, "p_active": pa})
            for kind in ("val", "val_late"):
                k = (val.kind == kind).to_numpy()
                a = k & np.isin(y, ACTIVE)
                rows.append({"repeat": rep, "model": vname, "kind": kind,
                             "active_F1": f1_score(y[a], np.array(ACTIVE)[pa[a].argmax(1)], labels=ACTIVE, average="macro"),
                             "F1": f1_score(y[k], pred[k], labels=LABELS, average="macro"),
                             "F1_50Hz": f1_score(y[k & ~sixty], pred[k & ~sixty], labels=LABELS, average="macro"),
                             "F1_60Hz": f1_score(y[k & sixty], pred[k & sixty], labels=LABELS, average="macro")})
    T = pd.DataFrame(rows)
    T.to_csv(out_dir / "summary.csv", index=False)
    for kind in ("val", "val_late"):
        x = T[T.kind == kind]
        print(f"\ngestures at {'0.7' if kind == 'val' else '0.9'} s, mean over repeats {args.repeats} (C={args.C}):")
        print(x.groupby("model", sort=False)[["active_F1", "F1", "F1_50Hz", "F1_60Hz"]].mean().round(3).to_string())
        base = x[x.model == "current (trees+TCN)"].set_index("repeat")
        print("change in six-class F1 vs current, per repeat:")
        for vname in [v for v in x.model.unique() if v.startswith("trees+TCN+linear")]:
            d = x[x.model == vname].set_index("repeat").F1 - base.F1
            print(f"  {vname:24s} {np.round(d.values, 3).tolist()}  mean {d.mean():+.3f}")
    print(f"\nsaved {out_dir} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
