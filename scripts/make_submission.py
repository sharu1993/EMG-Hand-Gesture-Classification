"""Build the Kaggle submission in one command.

    python scripts/make_submission.py            # ~15-20 min on Apple-silicon MPS
    python scripts/make_submission.py --retrain  # retrain the TCN even if a checkpoint exists

Steps (each in its own process: scikit-learn and PyTorch must not share one on macOS):
  1. train the Riemann TCN with aligned windows on all 28 sessions
     (train_tcn.py --final --input riemann --align -> models/aura_tcn_riemann_align.pt);
     skipped if the checkpoint already exists, unless --retrain
  2. TCN probabilities for the 348 test actions   (predict_test.py --model tcn)
  3. train the Riemann + IMU feature baseline on all sessions and score the
     test actions                                    (predict_test.py --model baseline)
  4. average the two models' probabilities with equal weights (the best
     cross-validated setup: macro F1 0.638 with sessions held out) and write
     outputs/submission.csv in sample_submission.csv's order and columns

The submission is checked before it is written: every test action exactly
once, no extra ids, labels from the training set only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "vr-calibration-free-emg-challenge-pieeg"
sys.path.insert(0, str(ROOT))
from emg_gestures.segments import LABELS  # noqa: E402


def run(step: str, cmd: list[str]) -> None:
    print(f"\n===== {step}: {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT)
    print(f"===== {step} finished in {(time.time() - t0) / 60:.1f} min (exit code {rc})", flush=True)
    if rc != 0:
        raise SystemExit(f"{step} failed")


def load_probs(path: Path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    if list(z["labels"]) != LABELS:
        raise SystemExit(f"{path}: unexpected label order {list(z['labels'])}")
    return pd.DataFrame(z["probs"], index=z["action_id"], columns=LABELS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain", action="store_true", help="retrain the TCN checkpoint")
    ap.add_argument("--device", default="mps", help="for TCN training (mps, cuda or cpu)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/aura_tcn_riemann_align.pt")
    ap.add_argument("--sample", type=Path, default=DATA / "sample_submission.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/submission.csv")
    ap.add_argument("--quick", action="store_true",
                    help="tiny training settings to check the pipeline runs (not a real submission)")
    args = ap.parse_args()

    py, t0 = sys.executable, time.time()
    quick_tcn = ["--per-session", "20", "--epochs", "1"] if args.quick else []
    quick_base = ["--per-session", "20"] if args.quick else []
    workers = ["--workers", str(args.workers)]

    if args.retrain or args.quick or not args.checkpoint.exists():
        run("1/3 train TCN on all sessions",
            [py, "scripts/train_tcn.py", "--final", "--input", "riemann", "--align",
             "--device", args.device, "--out", str(args.checkpoint), *workers, *quick_tcn])
    else:
        print(f"using existing checkpoint {args.checkpoint} (pass --retrain to rebuild it)")
    tcn_probs = ROOT / "outputs/test_probs_tcn.npz"
    base_probs = ROOT / "outputs/test_probs_baseline.npz"
    run("2/3 TCN test predictions",
        [py, "scripts/predict_test.py", "--model", "tcn", "--checkpoint", str(args.checkpoint),
         "--out", str(tcn_probs)])
    run("3/3 baseline training + test predictions",
        [py, "scripts/predict_test.py", "--model", "baseline", "--out", str(base_probs),
         *workers, *quick_base])

    # combine with equal weights and write in the sample submission's order
    sample = pd.read_csv(args.sample)
    id_col, label_col = sample.columns[:2]
    tcn, base = load_probs(tcn_probs), load_probs(base_probs)
    ids = sample[id_col].astype(str)
    missing = set(ids) - set(tcn.index) | set(ids) - set(base.index)
    extra = (set(tcn.index) | set(base.index)) - set(ids)
    if missing or extra:
        raise SystemExit(f"ids differ from {args.sample.name}: missing {sorted(missing)[:5]}, "
                         f"extra {sorted(extra)[:5]}")
    probs = (tcn.loc[ids].to_numpy() + base.loc[ids].to_numpy()) / 2
    sub = pd.DataFrame({id_col: ids.to_numpy(), label_col: np.array(LABELS)[probs.argmax(1)]})

    assert len(sub) == len(sample) and sub[id_col].is_unique
    assert (sub[id_col].to_numpy() == sample[id_col].astype(str).to_numpy()).all()
    assert set(sub[label_col]) <= set(LABELS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)

    agree = (tcn.loc[ids].to_numpy().argmax(1) == base.loc[ids].to_numpy().argmax(1)).mean()
    print(f"\nwrote {args.out}: {len(sub)} rows, columns {list(sub.columns)}")
    print("predicted classes:", sub[label_col].value_counts().reindex(LABELS, fill_value=0).to_dict())
    print(f"TCN and baseline agree on {agree:.0%} of test actions")
    if args.quick:
        print("NOTE: --quick used tiny training settings; this file is only a pipeline check.")
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
