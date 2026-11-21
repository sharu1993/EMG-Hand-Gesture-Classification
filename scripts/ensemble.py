"""Combine cross-validated predictions of several models by averaging their
class probabilities, and score the combination on the same windows.

Inputs are the *.probs.npz files written by scripts/train_tcn.py and
scripts/baseline.py. Both score every window with a model that never saw its
session, in the same window order; this script checks that the windows match
(sample id and offset) before combining.

    python scripts/ensemble.py outputs/tcn_frames.probs.npz outputs/baseline_emg_riemann_imu.probs.npz
    python scripts/ensemble.py a.probs.npz b.probs.npz c.probs.npz --geometric
    python scripts/ensemble.py --auto        # every probs file in outputs/: best TCN + best baseline

Reported for each single model and for the equal-weight average:
steady-window macro F1, 50 Hz / 60 Hz sessions, untrimmed 1.4 s windows.
With exactly two models it also shows a weight sweep. Weights picked from that
sweep are chosen on the same cross-validation data, so treat them as
optimistic; equal weights are the honest default.

NumPy only (no scikit-learn or PyTorch), so it runs in any environment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emg_gestures.metrics import macro_f1  # noqa: E402

SESSIONS_60HZ = {10, 11, 15, 16, 22, 23, 24}


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if np.isnan(d["prob_val"]).any():
        raise SystemExit(f"{path}: not every window was scored (partial --fold-ids run?)")
    return d


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


def score(p_val, p_raw, ref) -> dict:
    y, s = ref["y_val"], ref["session_val"]
    pred = p_val.argmax(1)
    g60 = np.isin(s, list(SESSIONS_60HZ))
    return {"steady_F1": macro_f1(y, pred), "50Hz": macro_f1(y[~g60], pred[~g60]),
            "60Hz": macro_f1(y[g60], pred[g60]),
            "untrimmed_1.4s": macro_f1(ref["y_raw"], p_raw.argmax(1))}


def combine(probs: list[np.ndarray], weights, geometric: bool) -> np.ndarray:
    w = np.asarray(weights, float) / np.sum(weights)
    if geometric:
        logp = sum(wi * np.log(np.clip(p, 1e-6, 1)) for wi, p in zip(w, probs))
        e = np.exp(logp - logp.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    return sum(wi * p for wi, p in zip(w, probs))


def auto_pick(outdir: Path) -> list[Path]:
    """Best TCN run and best baseline configuration by steady macro F1."""
    picks = []
    for prefix in ("tcn_", "baseline_"):
        cands = sorted(outdir.glob(f"{prefix}*.probs.npz"))
        best = None
        for c in cands:
            try:
                d = load(c)
            except SystemExit:
                continue
            f1 = macro_f1(d["y_val"], d["prob_val"].argmax(1))
            if best is None or f1 > best[0]:
                best = (f1, c)
        if best:
            picks.append(best[1])
    if len(picks) < 2:
        raise SystemExit(f"--auto needs at least one tcn_* and one baseline_* probs file in {outdir}")
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probs", type=Path, nargs="*", help="*.probs.npz files to combine")
    ap.add_argument("--auto", action="store_true",
                    help="pick the best tcn_* and best baseline_* file in --outdir")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="one weight per file (default: equal)")
    ap.add_argument("--geometric", action="store_true",
                    help="average log-probabilities instead of probabilities")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    paths = auto_pick(args.outdir) if args.auto else args.probs
    if len(paths) < 2:
        ap.error("give at least two probs files, or --auto")
    runs = {p.name.replace(".probs.npz", ""): load(p) for p in paths}
    check_same_windows(runs)
    ref = next(iter(runs.values()))
    weights = args.weights or [1.0] * len(runs)
    if len(weights) != len(runs):
        ap.error("--weights needs one value per file")

    rows = [{"model": n, **score(d["prob_val"], d["prob_raw"], ref)} for n, d in runs.items()]
    kind = "geometric" if args.geometric else "arithmetic"
    combo = {"model": f"combined ({kind} mean, weights {weights})",
             **score(combine([d["prob_val"] for d in runs.values()], weights, args.geometric),
                     combine([d["prob_raw"] for d in runs.values()], weights, args.geometric), ref)}
    table = pd.DataFrame(rows + [combo]).set_index("model")
    print("Macro F1, sessions held out (same windows for every model)")
    print(table.round(3).to_string())

    if len(runs) == 2:
        names = list(runs)
        sweep = []
        for w in np.linspace(0, 1, 11):
            pv = combine([runs[n]["prob_val"] for n in names], [w, 1 - w], args.geometric)
            pr = combine([runs[n]["prob_raw"] for n in names], [w, 1 - w], args.geometric)
            sweep.append({f"weight on {names[0]}": round(w, 1), **score(pv, pr, ref)})
        print("\nWeight sweep (chosen on the same CV data: optimistic, for orientation only)")
        print(pd.DataFrame(sweep).round(3).to_string(index=False))

    out = args.outdir / "ensemble.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
