"""Build the Kaggle submission in one command.

    python scripts/make_submission.py            # ~3-5 min on Apple-silicon MPS
    python scripts/make_submission.py --retrain  # retrain the TCN even if a checkpoint exists

Which two models: the pair that `ensemble.py --auto` ranked best in
cross-validation (outputs/ensemble_choice.json, written by scripts/run_all.py).
Without that file, or with --default-pair, the Riemann TCN with aligned
windows + the Riemann + IMU feature baseline with aligned windows.

Steps (each in its own process: scikit-learn and PyTorch must not share one on macOS):
  1. train the chosen TCN on all 28 sessions -> models/aura_tcn_<input>[_align].pt;
     skipped if that checkpoint already exists, unless --retrain (retrain after
     any change to preprocessing or training code)
  2. TCN probabilities for the 348 test actions   (predict_test.py --model tcn)
  3. train the chosen feature baseline on all sessions and score the test
     actions                                         (predict_test.py --model baseline)
  4. average the two models' probabilities with equal weights and write
     outputs/submission.csv in sample_submission.csv's order and columns

The submission is checked before it is written: every test action exactly
once, no extra ids, labels from the training set only.
"""
from __future__ import annotations

import argparse
import json
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

# (TCN input, TCN aligned, baseline feature set, baseline aligned) used when no
# usable outputs/ensemble_choice.json exists.
DEFAULT_PAIR = ("riemann", True, "Riemann + IMU", True)


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
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="TCN checkpoint (default models/aura_tcn_<input>[_align].pt)")
    ap.add_argument("--choice", type=Path, default=ROOT / "outputs/ensemble_choice.json",
                    help="model pair picked by `ensemble.py --auto` (written by run_all.py)")
    ap.add_argument("--default-pair", action="store_true",
                    help="ignore --choice and use the default pair (Riemann TCN with "
                         "alignment + Riemann + IMU baseline with alignment)")
    ap.add_argument("--sample", type=Path, default=DATA / "sample_submission.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/submission.csv")
    ap.add_argument("--quick", action="store_true",
                    help="tiny training settings to check the pipeline runs (not a real submission)")
    args = ap.parse_args()

    py, t0 = sys.executable, time.time()
    tcn_input, tcn_align, base_config, base_align = DEFAULT_PAIR
    if not args.default_pair and args.choice.exists():
        choice = json.loads(args.choice.read_text())
        if choice.get("usable_for_submission"):
            tcn_input, tcn_align = choice["tcn"]["input"], bool(choice["tcn"]["align"])
            base_config, base_align = choice["baseline"]["config"], bool(choice["baseline"]["align"])
            sc = choice.get("scores", {})
            print(f"using the pair from {args.choice.name} (ranked by {choice.get('rank_by')}): "
                  f"cross-validated balanced {sc.get('balanced')}, steady {sc.get('steady_F1')}, "
                  f"50 Hz {sc.get('50Hz')}, 60 Hz {sc.get('60Hz')}")
        else:
            print(f"{args.choice.name} is not usable ({'; '.join(choice.get('problems', []))}); "
                  f"using the default pair")
    else:
        print("using the default pair" + ("" if args.default_pair else
              f" ({args.choice.name} not found; run scripts/run_all.py to create it)"))
    print(f"  TCN: input {tcn_input}, align {tcn_align} | baseline: {base_config}, align {base_align}")
    if args.checkpoint is None:
        args.checkpoint = ROOT / f"models/aura_tcn_{tcn_input}{'_align' if tcn_align else ''}.pt"
    quick_tcn = ["--per-session", "20", "--epochs", "1"] if args.quick else []
    quick_base = ["--per-session", "20"] if args.quick else []
    workers = ["--workers", str(args.workers)]

    if args.retrain or args.quick or not args.checkpoint.exists():
        run("1/3 train TCN on all sessions",
            [py, "scripts/train_tcn.py", "--final", "--input", tcn_input,
             *(["--align"] if tcn_align else []),
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
         "--baseline-config", base_config,
         "--baseline-align" if base_align else "--no-baseline-align",
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
