"""Class probabilities for the competition test actions, from one model.

    python scripts/predict_test.py --model baseline     # Riemann + IMU gradient boosting
    python scripts/predict_test.py --model tcn          # needs models/aura_tcn_riemann_align.pt

--model baseline  trains the best feature baseline from cross-validation
                  (Riemann + IMU features, aligned training windows) on all 28
                  training sessions, then scores the test actions.
--model tcn       loads a checkpoint written by
                  `train_tcn.py --final --input riemann --align` and scores the
                  test actions with it (CPU is enough for 348 actions).

Writes outputs/test_probs_<model>.npz: action_id, probs (n, 6), labels.
Run each model in its own process (scripts/make_submission.py does): scikit-learn
and PyTorch must not share a process on macOS (OpenMP clash).
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see the note in scripts/train_tcn.py

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from emg_gestures.segments import LABELS  # noqa: E402
from emg_gestures.testset import load_test_actions  # noqa: E402


def predict_baseline(actions, args) -> np.ndarray:
    import pandas as pd
    from baseline import CONFIGS, META, feature_table, new_model  # scripts/baseline.py
    from emg_gestures.align import align_segments
    from emg_gestures.features import window_features
    from emg_gestures.segments import load_segments
    from emg_gestures.windows import ALIGNED, WindowSampler

    segs = align_segments(load_segments(args.segments))
    train_w = []
    for s, g in segs.groupby("session"):
        train_w += WindowSampler(g, trims=ALIGNED, seed=args.seed * 1000 + s).sample_many(args.per_session)
    train = feature_table(train_w, args.workers)
    keep = CONFIGS[args.baseline_config]
    cols = [c for c in train.columns if c not in META and keep(c)]
    clf = new_model(args.seed).fit(train[cols], train.label)
    test = pd.DataFrame([window_features(e, i, riemann=True) for _, e, i in actions])
    order = [list(clf.classes_).index(lab) for lab in LABELS]
    print(f"baseline '{args.baseline_config}': {len(cols)} features, {len(train)} training windows")
    return clf.predict_proba(test[cols])[:, order]


def predict_tcn(actions, args) -> np.ndarray:
    import torch
    from emg_gestures.frames import imu_frames, window_frames
    from emg_gestures.model import AuraTCN, preprocess
    from emg_gestures.riemann import riemann_frames

    ck = torch.load(args.checkpoint, map_location="cpu")
    kind = ck["input"]
    model = AuraTCN(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    feats = []
    for _, e, i in actions:
        e, i = preprocess(e, i)
        if kind == "frames":
            e, i = window_frames(e, i)
        elif kind == "riemann":
            e, i = riemann_frames(e), imu_frames(i)
        feats.append((e, i))
    T = max(len(e) for e, _ in feats)
    E = np.zeros((len(feats), T, feats[0][0].shape[1]), np.float32)
    I = np.zeros((len(feats), T, feats[0][1].shape[1]), np.float32)
    M = np.zeros((len(feats), T), bool)
    for k, (e, i) in enumerate(feats):
        E[k, :len(e)], I[k, :len(i)], M[k, :len(e)] = e, i, True
    E, I, M = torch.from_numpy(E), torch.from_numpy(I), torch.from_numpy(M)
    st = ck["standardizer"]
    if st:                                     # frames / riemann inputs
        E = (E - st["e_mu"]) / st["e_sd"]
        I = (I - st["i_mu"]) / st["i_sd"]
    mf = M.unsqueeze(-1).float()
    with torch.no_grad():
        probs = torch.softmax(model(E * mf, I * mf, M), 1).numpy()
    print(f"tcn '{args.checkpoint}': input {kind}, align {ck.get('align')}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    return probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["baseline", "tcn"], required=True)
    ap.add_argument("--test-dir", type=Path, default=ROOT / "vr-calibration-free-emg-challenge-pieeg/test")
    ap.add_argument("--segments", type=Path, default=ROOT / "data/train_segments.npz")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/aura_tcn_riemann_align.pt")
    ap.add_argument("--baseline-config", default="Riemann + IMU")
    ap.add_argument("--per-session", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    actions = load_test_actions(args.test_dir)
    print(f"{len(actions)} test actions loaded ({time.time() - t0:.0f} s)")
    probs = predict_baseline(actions, args) if args.model == "baseline" else predict_tcn(actions, args)
    out = args.out or ROOT / f"outputs/test_probs_{args.model}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, action_id=np.array([a for a, _, _ in actions]),
                        probs=probs.astype(np.float32), labels=np.array(LABELS))
    counts = np.bincount(probs.argmax(1), minlength=len(LABELS))
    print("predicted classes:", dict(zip(LABELS, counts.tolist())))
    print(f"saved {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
