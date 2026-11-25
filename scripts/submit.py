"""Train the two-stage model (quiet/active design) on all training windows and write a Kaggle submission.

    python3 scripts/submit.py --name quiet_active_tcn

The active branch averages the trees with temporal conv nets (TCNs) on the
channel-relationship frames (aura/sequence.py), 3 by default; --tcn-seeds 0
gives the trees alone. Cross-validation, 3 repeats, gestures at 0.7 s: six-class
macro F1 0.666 with the TCN vs 0.656 for the trees alone, better in every repeat.

Needs data/features/*.pkl (scripts/build_features.py) and, with TCNs,
data/features/frames_relations_{train,test}.npz (scripts/build_frames.py
--kind relations) and PyTorch (trains on the Apple GPU when available).
Writes runs/<name>/:
  submission.csv   action_id,gesture in sample_submission.csv order
  models.pkl       the trained tree models (one per seed), to reproduce or deploy
  tcn_active.pt    the trained TCNs: weights and input standardisation
  manifest.json    git commit, settings, checksums of inputs and outputs, class
                   counts, and the cross-validation scores if runs/<name>/metrics.json exists
Refuses to overwrite an existing submission unless --force, so a scored
submission can always be traced back to the code and models that made it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.io import LABELS  # noqa: E402
from aura.twostage import ACTIVE, TwoStage, average_parts, combine, proba6  # noqa: E402

SAMPLE = ROOT / "vr-calibration-free-emg-challenge-pieeg/sample_submission.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "uncommitted_changes": run("status", "--porcelain", "--untracked-files=no")}


def train_active_tcns(train: pd.DataFrame, feat: Path, args, t0: float) -> dict:
    """Train args.tcn_seeds TCNs on all active training windows (relation frames)
    and predict the test actions. Returns the averaged test probabilities (in
    ACTIVE order), what to save, and the device used."""
    from aura.sequence import pick_device, predict_tcn, train_tcn
    ftr, fte = np.load(feat / "frames_relations_train.npz"), np.load(feat / "frames_relations_test.npz")
    if len(ftr["X"]) != len(train):
        sys.exit("frames_relations_train.npz does not match train.pkl; rerun scripts/build_frames.py --kind relations")
    act = np.isin(train.label.to_numpy(), ACTIVE)
    y = np.array([ACTIVE.index(l) for l in train.label.to_numpy()[act]])
    device = pick_device(args.device)
    p, saved = np.zeros((len(fte["X"]), len(ACTIVE))), []
    for j in range(args.tcn_seeds):
        model, std = train_tcn(ftr["X"][act], ftr["M"][act], y, device, seed=1000 + j, epochs=args.tcn_epochs)
        p += predict_tcn(model, std, fte["X"], fte["M"], device) / args.tcn_seeds
        saved.append({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "n_feat": int(ftr["X"].shape[2]), "classes": ACTIVE,
                      "mu": std.mu, "sd": std.sd, "lo": std.lo})
        print(f"TCN {j + 1}/{args.tcn_seeds} trained on {device} ({time.time() - t0:.0f} s)", flush=True)
    return {"p_test": p, "saved": saved, "device": str(device)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="quiet_active")
    ap.add_argument("--seeds", type=int, default=3, help="models trained with different seeds, probabilities averaged")
    ap.add_argument("--no-gain-aug", action="store_true")
    ap.add_argument("--tcn-seeds", type=int, default=3, help="TCNs for the active branch, averaged; 0 = trees alone")
    ap.add_argument("--tcn-epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", help="for the TCNs: auto (Apple GPU if available), mps, cuda or cpu")
    ap.add_argument("--force", action="store_true", help="overwrite an existing submission in runs/<name>/")
    args = ap.parse_args()

    out = ROOT / "runs" / args.name
    if (out / "submission.csv").exists() and not args.force:
        sys.exit(f"{out / 'submission.csv'} exists; pick another --name or pass --force")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feat = ROOT / "data/features"
    train = pd.read_pickle(feat / "train.pkl")
    test = pd.read_pickle(feat / "test.pkl")

    models, parts = [], []
    for seed in range(args.seeds):
        m = TwoStage(seed=seed, gain_aug=not args.no_gain_aug).fit(train)
        parts.append(m.predict_parts(test))
        models.append(m)
        print(f"model {seed + 1}/{args.seeds} trained ({time.time() - t0:.0f} s)", flush=True)
    parts = average_parts(parts)
    tcn = None
    if args.tcn_seeds > 0:
        tcn = train_active_tcns(train, feat, args, t0)
        p_tcn = tcn.pop("p_test")
        test = test.assign(**{f"p_tcn_{g}": p_tcn[:, j] for j, g in enumerate(ACTIVE)},
                           **{f"p_trees_{g}": parts["p_active"][:, j] for j, g in enumerate(ACTIVE)})
        parts["p_active"] = (parts["p_active"] + p_tcn) / 2
    p6 = proba6(parts)
    test = test.assign(gesture=combine(parts), p_quiet=parts["p_quiet"], p_pinch_if_quiet=parts["p_pinch"],
                       **{f"p_{c}": p6[:, j] for j, c in enumerate(LABELS)})

    sample = pd.read_csv(SAMPLE)
    sub = sample[["action_id"]].merge(test[["action_id", "gesture"]], on="action_id", how="left")
    if sub.gesture.isna().any() or len(sub) != len(sample):
        sys.exit("test actions do not match sample_submission.csv")
    sub.to_csv(out / "submission.csv", index=False)
    test.drop(columns=[c for c in test.columns if c[:2] in ("A_", "B_", "S_")]).to_csv(out / "test_probabilities.csv", index=False)
    with open(out / "models.pkl", "wb") as f:
        pickle.dump(models, f)
    if tcn is not None:
        import torch
        torch.save(tcn["saved"], out / "tcn_active.pt")

    metrics = out / "metrics.json"
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_state(),
        "settings": {"seeds": args.seeds, "design": "quiet/active", "gain_aug": not args.no_gain_aug,
                     "features": len(models[0].cols),
                     "training_windows": len(train),
                     "active_tcn": None if tcn is None else {"seeds": args.tcn_seeds, "epochs": args.tcn_epochs,
                                                             "frames": "relations", "device": tcn["device"],
                                                             "averaged_with_trees": True}},
        "sha256": {"submission.csv": sha256(out / "submission.csv"), "models.pkl": sha256(out / "models.pkl"),
                   "train_features": sha256(feat / "train.pkl"), "test_features": sha256(feat / "test.pkl"),
                   **({} if tcn is None else {"tcn_active.pt": sha256(out / "tcn_active.pt"),
                                              "train_frames": sha256(feat / "frames_relations_train.npz"),
                                              "test_frames": sha256(feat / "frames_relations_test.npz")})},
        "predicted_classes": sub.gesture.value_counts().to_dict(),
        "cv": json.loads(metrics.read_text()) if metrics.exists() else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {out / 'submission.csv'}: {len(sub)} rows; predicted classes {manifest['predicted_classes']}")
    if manifest["git"]["uncommitted_changes"]:
        print("warning: uncommitted changes in tracked files; the manifest records them")
    print(f"commit {manifest['git']['commit'][:7]}; done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
