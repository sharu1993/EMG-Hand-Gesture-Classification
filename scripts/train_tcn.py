"""Train and cross-validate the dual-branch TCN (emg_gestures.model.AuraTCN).

Same protocol as scripts/baseline.py so the numbers are comparable:
  * training windows: WindowSampler, steady part of each segment only (no
    transitions between gestures), random length 0.75-1.2 s, class-balanced,
    drawn per session into a fixed pool and augmented on the fly
  * validation: session-held-out folds, fixed 1.2 s steady windows; also the
    untrimmed 1.4 s windows (test-like) and scores per setup (50 / 60 Hz)

Runs on the Apple-silicon GPU (MPS) by default; --device cpu to override.

    python scripts/train_tcn.py data/train_segments.npz                 # 7-fold CV
    python scripts/train_tcn.py data/train_segments.npz --folds 7 --fold-ids 0 5 6
    python scripts/train_tcn.py data/train_segments.npz --final --out models/aura_tcn.pt
"""
from __future__ import annotations

import os

# macOS: two OpenMP runtimes (libomp.dylib) end up in this process (PyTorch's
# own plus one linked by another installed package), which aborts with
# "OMP: Error #15". This tells the Intel/LLVM OpenMP runtime to tolerate the
# duplicate. Unsupported by the runtime's authors: it can in rare cases crash or
# give wrong numbers, so a clean environment (one libomp) is the better fix.
# Must be set before numpy/torch are imported. Override with
# KMP_DUPLICATE_LIB_OK=FALSE to see whether the environment is clean.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# No scikit-learn in this process: its OpenMP runtime clashes with PyTorch's on
# macOS (OMP Error #15). emg_gestures.metrics provides the pieces we need.
from emg_gestures.metrics import confusion, f1_per_class, macro_f1, session_folds  # noqa: E402
from emg_gestures.model import AuraTCN, count_parameters, preprocess  # noqa: E402
from emg_gestures.segments import LABELS, load_segments  # noqa: E402
from emg_gestures.windows import LABEL_TO_ID, NO_TRIM, WindowSampler, fixed_windows  # noqa: E402

SESSIONS_60HZ = {10, 11, 15, 16, 22, 23, 24}


def pick_device(name: str) -> torch.device:
    """Default: Apple-silicon GPU (MPS). Falls back to CUDA, then CPU, with a warning."""
    if name != "mps":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    reason = ("PyTorch was not built with MPS" if not torch.backends.mps.is_built()
              else "MPS is not available (needs macOS 12.3+ on Apple silicon)")
    fallback = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"WARNING: {reason}; using {fallback} instead", flush=True)
    return torch.device(fallback)


def _prep(w):
    e, i = preprocess(w.emg, w.imu)
    return e, i, LABEL_TO_ID[w.label], w.session


def build_pool(windows, workers: int) -> dict:
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_prep, windows, chunksize=128))
    return {"emg": [o[0] for o in out], "imu": [o[1] for o in out],
            "y": np.array([o[2] for o in out]), "session": np.array([o[3] for o in out])}


def collate(pool: dict, idx: np.ndarray, augment: bool, rng: np.random.Generator):
    T = max(len(pool["emg"][i]) for i in idx)
    B = len(idx)
    emg = np.zeros((B, T, 8), np.float32)
    imu = np.zeros((B, T, 6), np.float32)
    mask = np.zeros((B, T), bool)
    for b, i in enumerate(idx):
        e, m = pool["emg"][i], pool["imu"][i]
        if augment:
            e = e * np.exp(rng.normal(0, 0.15, 8)).astype(np.float32)        # contact gain per channel
            e = e + rng.normal(0, 0.05, e.shape).astype(np.float32)          # sensor noise
            if rng.random() < 0.15:                                          # one bad contact
                e = e.copy()
                e[:, rng.integers(8)] = 0.0
            m = m * np.exp(rng.normal(0, 0.1, 6)).astype(np.float32)
            m = m + rng.normal(0, 0.05, m.shape).astype(np.float32)
        n = len(e)
        emg[b, :n], imu[b, :n], mask[b, :n] = e, m, True
    return (torch.from_numpy(emg), torch.from_numpy(imu), torch.from_numpy(mask),
            torch.from_numpy(pool["y"][idx].astype(np.int64)))


def predict(model, pool, idx, device, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            e, i, m, _ = collate(pool, idx[s:s + batch], False, None)
            out.append(torch.softmax(model(e.to(device), i.to(device), m.to(device)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


def train_model(pool, idx, args, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AuraTCN(use_imu=not args.no_imu, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * int(np.ceil(len(idx) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    for _ in range(args.epochs):
        model.train()
        order = rng.permutation(idx)
        for s in range(0, len(order), args.batch):
            e, i, m, y = collate(pool, order[s:s + args.batch], True, rng)
            loss = F.cross_entropy(model(e.to(device), i.to(device), m.to(device)), y.to(device),
                                   label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", type=Path)
    ap.add_argument("--per-session", type=int, default=600, help="training windows per session")
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--max-folds", type=int, default=None, help="run only the first N folds")
    ap.add_argument("--fold-ids", type=int, nargs="+", default=None,
                    help="run only these fold indices (0-based), e.g. to cover both setups")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="mps", help="mps (default), cuda or cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="train on all sessions and save")
    ap.add_argument("--out", type=Path, default=Path("models/aura_tcn.pt"))
    ap.add_argument("--report", type=Path, default=Path("outputs/tcn_cv.json"))
    args = ap.parse_args()

    t0 = time.time()
    device = pick_device(args.device)
    segs = load_segments(args.segments)
    tr_w, va_w, raw_w = [], [], []
    for s, g in segs.groupby("session"):
        tr_w += WindowSampler(g, seed=args.seed * 1000 + s).sample_many(args.per_session)
        va_w += fixed_windows(g, length=1.2, stride=0.5)
        raw_w += fixed_windows(g, length=1.4, stride=0.5, trims=NO_TRIM)
    train, val, raw = (build_pool(w, args.workers) for w in (tr_w, va_w, raw_w))
    n_par = count_parameters(AuraTCN(use_imu=not args.no_imu))
    print(f"device {device}; {len(tr_w)} train / {len(va_w)} steady val / {len(raw_w)} untrimmed "
          f"windows; model {n_par:,} parameters; data ready in {time.time() - t0:.0f} s", flush=True)

    if args.final:
        model = train_model(train, np.arange(len(train["y"])), args, device, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}  # loadable without MPS
        torch.save({"state_dict": state, "labels": LABELS,
                    "config": {"use_imu": not args.no_imu, "dropout": args.dropout}}, args.out)
        print(f"saved {args.out} ({time.time() - t0:.0f} s)")
        return

    sessions = np.array(sorted(segs.session.unique()))
    prob_val = np.full((len(val["y"]), len(LABELS)), np.nan)
    prob_raw = np.full((len(raw["y"]), len(LABELS)), np.nan)
    folds = session_folds(sessions, args.folds)  # same folds as scripts/baseline.py
    run = args.fold_ids if args.fold_ids is not None else range(len(folds))[:args.max_folds]
    for k in run:
        held = set(folds[k].tolist())
        tr_idx = np.flatnonzero(~np.isin(train["session"], list(held)))
        model = train_model(train, tr_idx, args, device, args.seed + k)
        for pool, prob in ((val, prob_val), (raw, prob_raw)):
            idx = np.flatnonzero(np.isin(pool["session"], list(held)))
            prob[idx] = predict(model, pool, idx, device)
        idx = np.flatnonzero(np.isin(val["session"], list(held)))
        f1 = macro_f1(val["y"][idx], prob_val[idx].argmax(1))
        print(f"fold {k + 1}/{args.folds} sessions {sorted(held)}: steady macro F1 {f1:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    done = ~np.isnan(prob_val[:, 0])
    y, p = val["y"][done], prob_val[done].argmax(1)
    sess = val["session"][done]
    g60 = np.isin(sess, list(SESSIONS_60HZ))
    rdone = ~np.isnan(prob_raw[:, 0])
    report = {
        "steady_macro_F1": macro_f1(y, p),
        "50Hz_sessions": macro_f1(y[~g60], p[~g60]) if (~g60).any() else None,
        "60Hz_sessions": macro_f1(y[g60], p[g60]) if g60.any() else None,
        "untrimmed_1.4s_macro_F1": macro_f1(raw["y"][rdone], prob_raw[rdone].argmax(1)),
        "per_class_F1": dict(zip(LABELS, f1_per_class(y, p, range(6)).round(3).tolist())),
        "per_session_F1": {int(s): round(macro_f1(y[sess == s], p[sess == s]), 3)
                           for s in np.unique(sess)},
        "n_parameters": n_par, "args": {k: str(v) for k, v in vars(args).items()},
    }
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in report.items()
                      if k != "args"}, indent=1))
    print("Confusion matrix (rows = true):")
    print(pd.DataFrame(confusion(y, p, 6), index=LABELS, columns=LABELS))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))
    print(f"saved {args.report}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
