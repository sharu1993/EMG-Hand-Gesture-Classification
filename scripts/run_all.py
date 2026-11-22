"""Run the whole evaluation in one go:

    1. feature baseline with per-segment alignment   (scripts/baseline.py --align)
    2. TCN sweep: riemann / riemann + correlation x person-variation augmentation
       off / on                                      (scripts/train_tcn.py --sweep)
    3. ensemble of the best TCN and the best baseline (scripts/ensemble.py --auto)

    python scripts/run_all.py              # full run (~1 hour on Apple-silicon MPS)
    python scripts/run_all.py --seeds 3    # average 3 TCN seeds per fold (~3x the TCN time)
    python scripts/run_all.py --quick      # tiny settings, ~3 min: checks that everything runs

Each step runs in its own Python process (scikit-learn and PyTorch must not
share a process on macOS, see the OpenMP note in scripts/train_tcn.py), using
the same Python interpreter as this script. Output is shown live and also
written to outputs/run_all.log. Stops at the first failing step.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICK = {
    "baseline": ["--per-session", "20"],
    "tcn": ["--per-session", "20", "--epochs", "2"],
}


def run(step: str, cmd: list[str], log) -> None:
    banner = f"\n===== {step}: {' '.join(cmd[1:])}\n"
    print(banner, flush=True)
    log.write(banner)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
    proc.wait()
    msg = f"===== {step} finished in {(time.time() - t0) / 60:.1f} min (exit code {proc.returncode})\n"
    print(msg, flush=True)
    log.write(msg)
    log.flush()
    if proc.returncode != 0:
        raise SystemExit(f"{step} failed; see the output above and outputs/run_all.log")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny settings to check the pipeline runs")
    ap.add_argument("--device", default="mps", help="passed to the TCN step (mps, cuda or cpu)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=1,
                    help="TCN models per fold, averaged (the submission uses 3 regardless)")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    py = sys.executable
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    workers = ["--workers", str(args.workers)]
    steps = [
        ("1/3 feature baseline", [py, "scripts/baseline.py", "--align", "--outdir", str(outdir),
                                   *workers, *(QUICK["baseline"] if args.quick else [])]),
        ("2/3 TCN sweep", [py, "scripts/train_tcn.py", "--sweep", "--device", args.device,
                           "--report", str(outdir / "tcn_cv.json"), "--seeds", str(args.seeds),
                           *workers,
                           *(QUICK["tcn"] if args.quick else [])]),
        ("3/3 ensemble", [py, "scripts/ensemble.py", "--auto", "--outdir", str(outdir)]),
    ]
    t0 = time.time()
    with open(outdir / "run_all.log", "w") as log:
        log.write(f"python {py}\nquick {args.quick}\n")
        for step, cmd in steps:
            run(step, cmd, log)
    print(f"All done in {(time.time() - t0) / 60:.1f} min. Results: {outdir}/ "
          f"(tcn_sweep.csv, ensemble.csv, *.probs.npz, run_all.log)")


if __name__ == "__main__":
    main()
