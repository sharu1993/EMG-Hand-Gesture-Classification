"""Rebuild submission.csv from the saved weights, without training.

Run from the repository root; uses the copy of the code in code/ (commit c31c57d):

    python3 submissions/2026-11-25_quiet_active_tcn_rp_0.68375/predict.py

Inputs (defaults are the repository's own paths):
  data/features/test.pkl                    scripts/build_features.py
  data/features/frames_relations_test.npz   scripts/build_frames.py --kind relations
  weights/models.pkl, weights/tcn_active.pt, weights/tcn_rest_pinch.pt  (this folder)
Writes predicted.csv in this folder and reports whether it matches submission.csv.

The combination is the one in scripts/submit.py (lines 121-138):
trees averaged over 3 seeds; the active-branch probabilities averaged half and
half with the mean of 3 active TCNs; p(pinch | quiet) averaged half and half
with the mean of 3 rest-vs-pinch TCNs; fixed 0.5 gates (aura/twostage.py, combine).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "code"))      # the code as it was at commit c31c57d

from aura.sequence import Standardizer, build_tcn, pick_device, predict_tcn  # noqa: E402
from aura.twostage import ACTIVE, average_parts, combine  # noqa: E402


def load_tcns(path: Path, device):
    import torch
    out = []
    for s in torch.load(path, map_location="cpu", weights_only=False):
        model = build_tcn(n_feat=s["n_feat"], n_classes=len(s["classes"]))
        model.load_state_dict(s["state_dict"])
        std = Standardizer.__new__(Standardizer)
        std.mu, std.sd, std.lo = s["mu"], s["sd"], s["lo"]
        out.append((model.to(device), std, s["classes"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-features", default=str(ROOT / "data/features/test.pkl"))
    ap.add_argument("--test-frames", default=str(ROOT / "data/features/frames_relations_test.npz"))
    ap.add_argument("--weights", default=str(HERE / "weights"))
    ap.add_argument("--sample", default=str(ROOT / "vr-calibration-free-emg-challenge-pieeg/sample_submission.csv"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    w = Path(args.weights)
    test = pd.read_pickle(args.test_features)
    fr = np.load(args.test_frames)
    if len(fr["X"]) != len(test):
        sys.exit("test frames do not match the test features")
    device = pick_device(args.device)

    with open(w / "models.pkl", "rb") as f:
        trees = pickle.load(f)
    parts = average_parts([m.predict_parts(test) for m in trees])

    act = [predict_tcn(m, s, fr["X"], fr["M"], device) for m, s, c in load_tcns(w / "tcn_active.pt", device)]
    parts["p_active"] = (parts["p_active"] + np.mean(act, axis=0)) / 2

    rp = [predict_tcn(m, s, fr["X"], fr["M"], device)[:, c.index("pinch")]
          for m, s, c in load_tcns(w / "tcn_rest_pinch.pt", device)]
    parts["p_pinch"] = (parts["p_pinch"] + np.mean(rp, axis=0)) / 2

    pred = pd.DataFrame({"action_id": test.action_id.to_numpy(), "gesture": combine(parts)})
    sample = pd.read_csv(args.sample)
    sub = sample[["action_id"]].merge(pred, on="action_id", how="left")
    if sub.gesture.isna().any() or len(sub) != len(sample):
        sys.exit("test actions do not match sample_submission.csv")
    sub.to_csv(HERE / "predicted.csv", index=False)

    ref = HERE / "submission.csv"
    if ref.exists():
        r = pd.read_csv(ref)
        same = (r.action_id.to_numpy() == sub.action_id.to_numpy()).all()
        agree = (r.gesture.to_numpy() == sub.gesture.to_numpy()).mean() if same else float("nan")
        print(f"wrote {HERE / 'predicted.csv'}; agrees with submission.csv on {agree:.1%} of {len(sub)} actions")
    else:
        print(f"wrote {HERE / 'predicted.csv'}")


if __name__ == "__main__":
    main()
