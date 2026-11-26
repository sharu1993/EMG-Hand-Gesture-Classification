"""Train the two-stage model (quiet/active design) on all training windows and write a Kaggle submission.

    python3 scripts/submit.py --name quiet_active_tcn                                   # pick 1
    python3 scripts/submit.py --name quiet_active_tcn_rp --tcn-rest-pinch 3             # 0.68375
    python3 scripts/submit.py --name car_tcn_rp --tcn-rest-pinch 3 --car                # + CAR features

The active branch averages the trees with temporal conv nets (TCNs) on the
channel-relationship frames (aura/sequence.py), 3 by default; --tcn-seeds 0
gives the trees alone. Cross-validation, 3 repeats, gestures at 0.7 s: six-class
macro F1 0.666 with the TCN vs 0.656 for the trees alone, better in every repeat.
--tcn-rest-pinch N also averages N rest-vs-pinch TCNs with the trees in the quiet
branch: 0.669 in cross-validation, but worse than without them in one of 3 repeats.
--car also gives the common-average-reference features (data/features/car_*.pkl,
scripts/build_car_features.py) to stage 1 and the rest-vs-pinch trees, not the
active branch: full model 0.686 at 0.7 s and 0.680 at 0.9 s in cross-validation,
+0.017 and +0.020, better in every repeat (25 Nov 2026).

Needs data/features/*.pkl (scripts/build_features.py) and, with TCNs,
data/features/frames_relations_{train,test}.npz (scripts/build_frames.py
--kind relations) and PyTorch (trains on the Apple GPU when available).
Writes runs/<name>/:
  submission.csv   action_id,gesture in sample_submission.csv order
  models.pkl       the trained tree models (one per seed), to reproduce or deploy
  tcn_active.pt    the trained active-gesture TCNs: weights and input standardisation
  tcn_rest_pinch.pt  the trained rest-vs-pinch TCNs (with --tcn-rest-pinch)
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


def train_tcns(task: str, n: int, train: pd.DataFrame, feat: Path, args, t0: float) -> dict:
    """Train n TCNs on the relation frames and predict the test actions.

    task "active":     the four active gestures, on all active training windows;
                       returns (n_test, 4) probabilities in ACTIVE order.
    task "rest_pinch": rest vs pinch, on all pinch windows and an equal random
                       subset of rest windows (a different subset per TCN), as
                       the tree model; returns p(pinch) (n_test,).
    Also returns what to save and the device used."""
    from aura.sequence import pick_device, predict_tcn, train_tcn
    ftr, fte = np.load(feat / "frames_relations_train.npz"), np.load(feat / "frames_relations_test.npz")
    if len(ftr["X"]) != len(train):
        sys.exit("frames_relations_train.npz does not match train.pkl; rerun scripts/build_frames.py --kind relations")
    labels = train.label.to_numpy()
    device = pick_device(args.device)
    classes = ACTIVE if task == "active" else ["rest", "pinch"]
    p, saved = 0.0, []
    for j in range(n):
        seed = (1000 if task == "active" else 2000) + j
        if task == "active":
            idx = np.flatnonzero(np.isin(labels, ACTIVE))
            y = np.array([ACTIVE.index(l) for l in labels[idx]])
        else:
            ip = np.flatnonzero(labels == "pinch")
            ir = np.random.default_rng(seed).choice(np.flatnonzero(labels == "rest"), size=len(ip), replace=False)
            idx = np.r_[ip, ir]
            y = (labels[idx] == "pinch").astype(int)
        model, std = train_tcn(ftr["X"][idx], ftr["M"][idx], y, device, seed=seed, epochs=args.tcn_epochs)
        pr = predict_tcn(model, std, fte["X"], fte["M"], device)
        p = p + (pr if task == "active" else pr[:, 1]) / n
        saved.append({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "n_feat": int(ftr["X"].shape[2]), "classes": classes, "seed": seed,
                      "mu": std.mu, "sd": std.sd, "lo": std.lo})
        print(f"{task} TCN {j + 1}/{n} trained on {device} ({time.time() - t0:.0f} s)", flush=True)
    return {"p_test": p, "saved": saved, "device": str(device)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="quiet_active")
    ap.add_argument("--seeds", type=int, default=3, help="models trained with different seeds, probabilities averaged")
    ap.add_argument("--no-gain-aug", action="store_true")
    ap.add_argument("--tcn-seeds", type=int, default=3, help="TCNs for the active branch, averaged; 0 = trees alone")
    ap.add_argument("--tcn-rest-pinch", type=int, default=0,
                    help="rest-vs-pinch TCNs averaged with the trees in the quiet branch; 0 = trees alone")
    ap.add_argument("--tcn-epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", help="for the TCNs: auto (Apple GPU if available), mps, cuda or cpu")
    ap.add_argument("--car", action="store_true",
                    help="add the common-average-reference features to stage 1 and the rest-vs-pinch trees")
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
    extra, extra_for = (), ()
    if args.car:
        ctr, cte = pd.read_pickle(feat / "car_train.pkl"), pd.read_pickle(feat / "car_test.pkl")
        meta = ["session", "seg", "label", "kind", "offset", "n_rows"]
        if len(ctr) != len(train) or not (ctr[meta].to_numpy() == train[meta].to_numpy()).all():
            sys.exit("car_train.pkl does not match train.pkl; rerun scripts/build_car_features.py")
        if len(cte) != len(test) or not (cte[["action_id", "n_rows"]].to_numpy() == test[["action_id", "n_rows"]].to_numpy()).all():
            sys.exit("car_test.pkl does not match test.pkl; rerun scripts/build_car_features.py")
        cols = [c for c in ctr.columns if c.startswith("C_")]
        train = pd.concat([train, ctr[cols]], axis=1)
        test = pd.concat([test, cte[cols]], axis=1)
        extra, extra_for = ("C_",), ("quiet", "pinch")
        print(f"added {len(cols)} common-average-reference features to stage 1 and rest vs pinch", flush=True)

    models, parts = [], []
    for seed in range(args.seeds):
        m = TwoStage(seed=seed, gain_aug=not args.no_gain_aug, extra=extra, extra_for=extra_for).fit(train)
        parts.append(m.predict_parts(test))
        models.append(m)
        print(f"model {seed + 1}/{args.seeds} trained ({time.time() - t0:.0f} s)", flush=True)
    parts = average_parts(parts)
    tcn = rp = None
    if args.tcn_seeds > 0:
        tcn = train_tcns("active", args.tcn_seeds, train, feat, args, t0)
        p_tcn = tcn.pop("p_test")
        test = test.assign(**{f"p_tcn_{g}": p_tcn[:, j] for j, g in enumerate(ACTIVE)},
                           **{f"p_trees_{g}": parts["p_active"][:, j] for j, g in enumerate(ACTIVE)})
        parts["p_active"] = (parts["p_active"] + p_tcn) / 2
    if args.tcn_rest_pinch > 0:
        rp = train_tcns("rest_pinch", args.tcn_rest_pinch, train, feat, args, t0)
        p_rp = rp.pop("p_test")
        test = test.assign(p_pinch_if_quiet_tcn=p_rp, p_pinch_if_quiet_trees=parts["p_pinch"])
        parts["p_pinch"] = (parts["p_pinch"] + p_rp) / 2
    p6 = proba6(parts)
    test = test.assign(gesture=combine(parts), p_quiet=parts["p_quiet"], p_pinch_if_quiet=parts["p_pinch"],
                       **{f"p_{c}": p6[:, j] for j, c in enumerate(LABELS)})

    sample = pd.read_csv(SAMPLE)
    sub = sample[["action_id"]].merge(test[["action_id", "gesture"]], on="action_id", how="left")
    if sub.gesture.isna().any() or len(sub) != len(sample):
        sys.exit("test actions do not match sample_submission.csv")
    sub.to_csv(out / "submission.csv", index=False)
    test.drop(columns=[c for c in test.columns if c[:2] in ("A_", "B_", "S_", "C_")]).to_csv(out / "test_probabilities.csv", index=False)
    with open(out / "models.pkl", "wb") as f:
        pickle.dump(models, f)
    if tcn is not None or rp is not None:
        import torch
    if tcn is not None:
        torch.save(tcn["saved"], out / "tcn_active.pt")
    if rp is not None:
        torch.save(rp["saved"], out / "tcn_rest_pinch.pt")

    metrics = out / "metrics.json"
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_state(),
        "settings": {"seeds": args.seeds, "design": "quiet/active", "gain_aug": not args.no_gain_aug,
                     "features": len(models[0].cols),
                     "extra_features": None if not extra else {"prefixes": list(extra), "models": list(extra_for),
                                                               "count": len(cols)},
                     "training_windows": len(train),
                     "active_tcn": None if tcn is None else {"seeds": args.tcn_seeds, "epochs": args.tcn_epochs,
                                                             "frames": "relations", "device": tcn["device"],
                                                             "averaged_with_trees": True},
                     "rest_pinch_tcn": None if rp is None else {"seeds": args.tcn_rest_pinch, "epochs": args.tcn_epochs,
                                                                "frames": "relations", "device": rp["device"],
                                                                "averaged_with_trees": True}},
        "sha256": {"submission.csv": sha256(out / "submission.csv"), "models.pkl": sha256(out / "models.pkl"),
                   "train_features": sha256(feat / "train.pkl"), "test_features": sha256(feat / "test.pkl"),
                   **({} if not extra else {"car_train": sha256(feat / "car_train.pkl"),
                                            "car_test": sha256(feat / "car_test.pkl")}),
                   **({} if tcn is None else {"tcn_active.pt": sha256(out / "tcn_active.pt")}),
                   **({} if rp is None else {"tcn_rest_pinch.pt": sha256(out / "tcn_rest_pinch.pt")}),
                   **({} if tcn is None and rp is None else
                      {"train_frames": sha256(feat / "frames_relations_train.npz"),
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
