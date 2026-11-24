"""Combine cross-validated predictions of several models by averaging their
class probabilities, and score the combination on the same windows.

Inputs are the *.probs.npz files written by scripts/train_tcn.py and
scripts/baseline.py. Both score every window with a model that never saw its
session, in the same window order; this script checks that the windows match
(sample id and offset) before combining.

    python scripts/ensemble.py --auto        # every TCN x trees pair, with and without an MLP
    python scripts/ensemble.py --auto --rank-by steady
    python scripts/ensemble.py outputs/tcn_riemann.probs.npz outputs/baseline_riemann_imu_align.probs.npz

--auto  pairs every tcn_* file with every baseline_* (trees) file in --outdir,
        and also tries every such pair plus each mlp_* file (three models).
        Scores each equal-weight average, ranks them and reports the best in
        detail. Files from an older run whose validation windows differ from
        the newest files (e.g. before a preprocessing change) are skipped. The
        chosen combination is written to <outdir>/ensemble_choice.json, which
        scripts/make_submission.py reads to build the submission from the
        same models.

--shift-aug none|small|wide|any
        --auto only: consider only models trained with this patch-shift
        augmentation level (default none), so augmented and plain models are
        never mixed in one ensemble. "any" disables the filter.

--rank-by  testlike (default): macro F1 on test-like validation windows (1.4 s
           from 0.5 s after the cue, as the test actions); files without them
           (older runs) are skipped by --auto.
           balanced: mean of the 50 Hz and 60 Hz session scores;
           the test set is about half 50 Hz (test_1, 2, 4) and half 60 Hz
           (test_3, 5), so this matches it better than the overall score,
           which is dominated by the 21 training sessions at 50 Hz.
           steady / untrimmed: overall steady-window or untrimmed 1.4 s score.

Picking the best of several pairs on the same cross-validation data is
slightly optimistic; pairs within ~0.01 of each other should be treated as tied.

Reported for each single model and for the combination: steady-window macro F1,
50 Hz / 60 Hz sessions, balanced (their mean), untrimmed 1.4 s windows. With
exactly two models it also shows a weight sweep (orientation only; equal
weights are the honest default).

NumPy only (no scikit-learn or PyTorch), so it runs in any environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emg_gestures.metrics import macro_f1  # noqa: E402

SESSIONS_60HZ = {10, 11, 15, 16, 22, 23, 24}
RANK_KEYS = {"testlike": "testlike", "balanced": "balanced", "steady": "steady_F1",
             "untrimmed": "untrimmed_1.4s"}
# baseline.py file slugs -> its feature-set names (CONFIGS / ABLATIONS)
BASELINE_CONFIGS = {"emg_imu": "EMG + IMU", "riemann_imu": "Riemann + IMU",
                    "riemann_corr_imu": "Riemann + Corr + IMU",
                    "emg_riemann_imu": "EMG + Riemann + IMU", "emg": "EMG",
                    "emg_gain": "EMG + gain", "imu": "IMU"}


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if np.isnan(d["prob_val"]).any():
        raise SystemExit(f"{path}: not every window was scored (partial --fold-ids run?)")
    return d


def window_signature(d: dict) -> str:
    """Identifies the validation windows a file was scored on."""
    h = hashlib.sha1()
    for k in ("y_val", "session_val", "y_raw", "session_raw", "y_tl", "session_tl"):
        if k in d:                                     # test-like windows: newer runs only
            h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def check_same_windows(runs: dict[str, dict]) -> None:
    names = list(runs)
    ref = runs[names[0]]
    for n in names[1:]:
        d = runs[n]
        for part in ("val", "raw"):
            same = (len(d[f"y_{part}"]) == len(ref[f"y_{part}"])
                    and (d[f"y_{part}"] == ref[f"y_{part}"]).all()
                    and (d[f"session_{part}"] == ref[f"session_{part}"]).all())
            if same and f"sample_id_{part}" in d and f"sample_id_{part}" in ref:
                same = ((d[f"sample_id_{part}"] == ref[f"sample_id_{part}"]).all()
                        and np.allclose(d[f"offset_{part}"], ref[f"offset_{part}"], atol=1e-4))
            if not same:
                raise SystemExit(f"{n} and {names[0]} were scored on different {part} windows")


def score(p_val, p_raw, ref, p_tl=None) -> dict:
    y, s = ref["y_val"], ref["session_val"]
    pred = p_val.argmax(1)
    g60 = np.isin(s, list(SESSIONS_60HZ))
    f50, f60 = macro_f1(y[~g60], pred[~g60]), macro_f1(y[g60], pred[g60])
    return {"steady_F1": macro_f1(y, pred), "50Hz": f50, "60Hz": f60,
            "balanced": (f50 + f60) / 2,
            "untrimmed_1.4s": macro_f1(ref["y_raw"], p_raw.argmax(1)),
            "testlike": (macro_f1(ref["y_tl"], p_tl.argmax(1)) if p_tl is not None and "y_tl" in ref
                         else np.nan)}


def combine_tl(ds: list[dict], weights, geometric: bool):
    """Combined test-like probabilities, or None if any model lacks them."""
    if not all("prob_tl" in d for d in ds):
        return None
    return combine([d["prob_tl"] for d in ds], weights, geometric)


def combine(probs: list[np.ndarray], weights, geometric: bool) -> np.ndarray:
    w = np.asarray(weights, float) / np.sum(weights)
    if geometric:
        logp = sum(wi * np.log(np.clip(p, 1e-6, 1)) for wi, p in zip(w, probs))
        e = np.exp(logp - logp.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    return sum(wi * p for wi, p in zip(w, probs))


def model_info(path: Path) -> dict:
    """What it takes to retrain this model: TCN input/alignment from its report
    (tcn_<run>.json), baseline (trees) / MLP feature set and alignment from the
    file name."""
    name = path.name.replace(".probs.npz", "")
    if name.startswith("tcn_"):
        report = path.with_name(f"{name}.json")
        args = json.loads(report.read_text())["args"] if report.exists() else {}
        extras = {k: args.get(k) for k in ("group_balance", "adversarial") if k in args}
        return {"kind": "tcn", "name": name, "input": args.get("input"),
                "align": args.get("align") == "True",
                "shift_aug": args.get("shift_aug", "none"),         # absent in older reports: none
                "train_windows": args.get("train_windows", "steady"),
                "person_aug": args.get("person_aug") == "True",   # absent in older reports: off
                "extras_default": extras in ({}, {"group_balance": "False", "adversarial": "none"}),
                "report": report.name if report.exists() else None}
    kind, slug = name.split("_", 1)            # baseline_<slug>[_align][_win<set>][_shift<level>]
    shift = "none"
    for level in ("small", "wide"):
        if slug.endswith(f"_shift{level}"):
            shift, slug = level, slug[: -len(f"_shift{level}")]
    windows = "steady"
    for wset in ("both_natural", "testlike", "both"):
        if slug.endswith(f"_win{wset}"):
            windows, slug = wset, slug[: -len(f"_win{wset}")]
            break
    align = slug.endswith("_align")
    slug = slug[: -len("_align")] if align else slug
    return {"kind": kind, "name": name, "config": BASELINE_CONFIGS.get(slug), "align": align,
            "shift_aug": shift, "train_windows": windows}


def auto_candidates(outdir: Path, shift_aug: str = "none"):
    """Probability files that share the newest file's validation windows (and, unless
    shift_aug is "any", were trained with that patch-shift augmentation level)."""
    files = [p for p in sorted(outdir.glob("*.probs.npz"))
             if p.name.startswith(("tcn_", "baseline_", "mlp_"))]
    if shift_aug != "any":
        files = [p for p in files if model_info(p).get("shift_aug", "none") == shift_aug]
    loaded = {}
    for p in files:
        try:
            loaded[p] = load(p)
        except SystemExit as e:
            print(f"skipped {p.name}: {e}")
    if not loaded:
        raise SystemExit(f"no tcn_*/baseline_* probs files in {outdir}")
    newest = max(loaded, key=lambda p: p.stat().st_mtime)
    sig = window_signature(loaded[newest])
    keep = {p: d for p, d in loaded.items() if window_signature(d) == sig}
    for p in loaded:
        if p not in keep:
            print(f"skipped {p.name}: scored on different validation windows (older run)")
    tcn = {p: d for p, d in keep.items() if p.name.startswith("tcn_")}
    base = {p: d for p, d in keep.items() if p.name.startswith("baseline_")}
    mlp = {p: d for p, d in keep.items() if p.name.startswith("mlp_")}
    if not tcn or not base:
        raise SystemExit(f"--auto needs at least one current tcn_* and baseline_* file in {outdir}")
    return tcn, base, mlp


def rank_combinations(tcn: dict, base: dict, mlp: dict, rank_key: str) -> pd.DataFrame:
    """Every TCN x trees pair, and every pair plus one MLP; equal weights."""
    rows = []
    for (pt, dt), (pb, db), (pm, dm) in product(tcn.items(), base.items(),
                                               [(None, None)] + list(mlp.items())):
        ds = [dt, db] + ([dm] if dm is not None else [])
        s = score(combine([d["prob_val"] for d in ds], [1] * len(ds), False),
                  combine([d["prob_raw"] for d in ds], [1] * len(ds), False), dt,
                  combine_tl(ds, [1] * len(ds), False))
        rows.append({"tcn": pt.name.replace(".probs.npz", ""),
                     "baseline": pb.name.replace(".probs.npz", ""),
                     "mlp": pm.name.replace(".probs.npz", "") if pm else "-", **s,
                     "_paths": [pt, pb] + ([pm] if pm else [])})
    return pd.DataFrame(rows).sort_values(rank_key, ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probs", type=Path, nargs="*", help="*.probs.npz files to combine")
    ap.add_argument("--auto", action="store_true",
                    help="try every tcn_* x baseline_* pair in --outdir and pick the best")
    ap.add_argument("--rank-by", choices=list(RANK_KEYS), default="testlike")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="one weight per file (default: equal)")
    ap.add_argument("--geometric", action="store_true",
                    help="average log-probabilities instead of probabilities")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--shift-aug", choices=["none", "small", "wide", "any"], default="none",
                    help="--auto: only models trained with this shift augmentation level")
    args = ap.parse_args()
    rank_key = RANK_KEYS[args.rank_by]

    if args.auto:
        tcn, base, mlp = auto_candidates(args.outdir, args.shift_aug)
        ranked = rank_combinations(tcn, base, mlp, rank_key)
        show = ranked.drop(columns=["_paths"])
        top = 30
        print(f"{len(ranked)} combinations (TCN x trees, with and without an MLP), equal weights, "
              f"ranked by {args.rank_by}; top {min(top, len(ranked))} shown (sessions held out; "
              f"best-of-N on the same data is slightly optimistic, rows within ~0.01 are "
              f"effectively tied)")
        print(show.head(top).round(3).to_string(index=True))
        pairs_only = ranked[ranked.mlp == "-"]
        if len(pairs_only) and len(pairs_only) < len(ranked):
            print(f"best without an MLP: {pairs_only.iloc[0]['tcn']} + {pairs_only.iloc[0]['baseline']} "
                  f"({args.rank_by} {pairs_only.iloc[0][rank_key]:.3f})")
        best = ranked.iloc[0]
        paths = best["_paths"]
    else:
        paths = args.probs
    if len(paths) < 2:
        ap.error("give at least two probs files, or --auto")

    runs = {Path(p).name.replace(".probs.npz", ""): load(Path(p)) for p in paths}
    check_same_windows(runs)
    ref = next(iter(runs.values()))
    weights = args.weights or [1.0] * len(runs)
    if len(weights) != len(runs):
        ap.error("--weights needs one value per file")

    rows = [{"model": n, **score(d["prob_val"], d["prob_raw"], ref, d.get("prob_tl"))}
            for n, d in runs.items()]
    kind = "geometric" if args.geometric else "arithmetic"
    combo = {"model": f"combined ({kind} mean, weights {weights})",
             **score(combine([d["prob_val"] for d in runs.values()], weights, args.geometric),
                     combine([d["prob_raw"] for d in runs.values()], weights, args.geometric), ref,
                     combine_tl(list(runs.values()), weights, args.geometric))}
    table = pd.DataFrame(rows + [combo]).set_index("model")
    print(f"\n{'Chosen pair' if args.auto else 'Models'}: macro F1, sessions held out "
          f"(same windows for every model)")
    print(table.round(3).to_string())

    if len(runs) == 2:
        names = list(runs)
        sweep = []
        for w in np.linspace(0, 1, 11):
            pv = combine([runs[n]["prob_val"] for n in names], [w, 1 - w], args.geometric)
            pr = combine([runs[n]["prob_raw"] for n in names], [w, 1 - w], args.geometric)
            pt = combine_tl([runs[n] for n in names], [w, 1 - w], args.geometric)
            sweep.append({f"weight on {names[0]}": round(w, 1), **score(pv, pr, ref, pt)})
        print("\nWeight sweep (chosen on the same CV data: optimistic, for orientation only)")
        print(pd.DataFrame(sweep).round(3).to_string(index=False))

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "ensemble.csv"
    table.to_csv(out)
    print(f"\nsaved {out}")

    if args.auto:
        info = [model_info(Path(p)) for p in paths]
        choice = {"rank_by": args.rank_by, "tcn": info[0], "baseline": info[1],
                  "mlp": info[2] if len(info) > 2 else None,
                  "scores": {k: round(float(v), 4) for k, v in combo.items() if k != "model"}}
        problems = []
        if info[0].get("input") is None:
            problems.append(f"no report found for {info[0]['name']} (input type unknown)")
        if not info[0].get("extras_default", True):
            problems.append(f"{info[0]['name']} used --group-balance/--adversarial, which "
                            f"make_submission.py does not reproduce")
        for i in info[1:]:
            if i.get("config") is None:
                problems.append(f"unknown feature set in {i['name']}")
        choice["usable_for_submission"] = not problems
        choice["problems"] = problems
        path = args.outdir / "ensemble_choice.json"
        path.write_text(json.dumps(choice, indent=1, default=str))
        print(f"saved {path}: TCN {info[0]['name']} (input {info[0].get('input')}, "
              f"align {info[0].get('align')}, person aug {info[0].get('person_aug')}) + baseline {info[1]['name']} "
              f"({info[1].get('config')}, align {info[1].get('align')})"
              + (f" + MLP {info[2]['name']} ({info[2].get('config')}, align {info[2].get('align')})"
                 if len(info) > 2 else ""))
        for p in problems:
            print(f"WARNING: {p}; make_submission.py will fall back to its default pair")


if __name__ == "__main__":
    main()
