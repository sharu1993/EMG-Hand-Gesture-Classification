"""Train the two-stage model on all training windows and write a Kaggle submission.

    python scripts/submit.py --name two_stage_baseline

Needs data/features/*.pkl (scripts/build_features.py). Writes runs/<name>/:
  submission.csv   action_id,gesture in sample_submission.csv order
  models.pkl       the trained models (one per seed), to reproduce or deploy
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
from aura.io import GESTURES  # noqa: E402
from aura.twostage import TwoStage, combine  # noqa: E402

SAMPLE = ROOT / "vr-calibration-free-emg-challenge-pieeg/sample_submission.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "uncommitted_changes": run("status", "--porcelain", "--untracked-files=no")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="two_stage_baseline")
    ap.add_argument("--seeds", type=int, default=3, help="models trained with different seeds, probabilities averaged")
    ap.add_argument("--rule", choices=["gate", "soft"], default="gate")
    ap.add_argument("--no-gain-aug", action="store_true")
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

    models, p1s, p2s = [], [], []
    for seed in range(args.seeds):
        m = TwoStage(seed=seed, gain_aug=not args.no_gain_aug).fit(train)
        p1, p2 = m.predict_parts(test)
        models.append(m)
        p1s.append(p1)
        p2s.append(p2)
        print(f"model {seed + 1}/{args.seeds} trained ({time.time() - t0:.0f} s)", flush=True)
    p1, p2 = np.mean(p1s, 0), np.mean(p2s, 0)
    test = test.assign(gesture=combine(p1, p2, args.rule), p_gesture=p1,
                       **{f"p_{g}": p2[:, j] for j, g in enumerate(GESTURES)})

    sample = pd.read_csv(SAMPLE)
    sub = sample[["action_id"]].merge(test[["action_id", "gesture"]], on="action_id", how="left")
    if sub.gesture.isna().any() or len(sub) != len(sample):
        sys.exit("test actions do not match sample_submission.csv")
    sub.to_csv(out / "submission.csv", index=False)
    test.drop(columns=[c for c in test.columns if c[:2] in ("A_", "B_", "S_")]).to_csv(out / "test_probabilities.csv", index=False)
    with open(out / "models.pkl", "wb") as f:
        pickle.dump(models, f)

    metrics = out / "metrics.json"
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_state(),
        "settings": {"seeds": args.seeds, "rule": args.rule, "gain_aug": not args.no_gain_aug,
                     "stage1_features": len(models[0].c1), "stage2_features": len(models[0].c2),
                     "training_windows": len(train)},
        "sha256": {"submission.csv": sha256(out / "submission.csv"), "models.pkl": sha256(out / "models.pkl"),
                   "train_features": sha256(feat / "train.pkl"), "test_features": sha256(feat / "test.pkl")},
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
