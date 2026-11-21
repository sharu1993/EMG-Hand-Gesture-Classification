"""Train and cross-validate the dual-branch TCN (emg_gestures.model.AuraTCN).

Same protocol as scripts/baseline.py so the numbers are comparable:
  * training windows: WindowSampler, steady part of each segment only (no
    transitions between gestures), random length 0.75-1.2 s, class-balanced,
    drawn per session into a fixed pool and augmented on the fly
  * validation: session-held-out folds (same folds as the baseline), fixed
    1.2 s steady windows; also the untrimmed 1.4 s windows (test-like) and
    scores per recording setup (50 / 60 Hz mains)

Inputs (--input):
  frames  short-time features per 200 ms frame: log band power per channel,
          cross-channel correlations, IMU stats (emg_gestures/frames.py). Default.
  raw     the preprocessed signal itself (the first TCN version)

Speed: every window is preprocessed once and the whole pool sits on the device
as padded tensors; batches are index slices and augmentation runs in torch, so
there is no per-sample Python work in the training loop.

Second setup (60 Hz sessions):
  --group-balance           weight the loss so 50 Hz and 60 Hz sessions count equally
  --adversarial setup|session
                            domain-adversarial training: a gradient-reversal head
                            tries to tell the setup (or session) from the
                            embedding, pushing the embedding to drop that information
  --adv-weight 0.3          strength of the reversed gradient

Runs on the Apple-silicon GPU (MPS) by default; --device cpu to override.

    python scripts/train_tcn.py --sweep          # all four configurations, one comparison table
    python scripts/train_tcn.py data/train_segments.npz                       # frames, 7-fold CV
    python scripts/train_tcn.py data/train_segments.npz --group-balance --adversarial session
    python scripts/train_tcn.py data/train_segments.npz --input raw --fold-ids 0 5 6
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
import math
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
from emg_gestures.frames import (BAND_COLS_OF_CHANNEL, CORR_COLS_OF_CHANNEL,  # noqa: E402
                                 EMG_DIM, window_frames)
from emg_gestures.metrics import confusion, f1_per_class, macro_f1, session_folds  # noqa: E402
from emg_gestures.model import FRAME_CONFIG, RAW_CONFIG, AuraTCN, count_parameters, preprocess  # noqa: E402
from emg_gestures.segments import LABELS, load_segments  # noqa: E402
from emg_gestures.windows import LABEL_TO_ID, NO_TRIM, WindowSampler, fixed_windows  # noqa: E402

SESSIONS_60HZ = {10, 11, 15, 16, 22, 23, 24}

# --sweep: configurations run back to back on the same prepared data.
SWEEP = {
    "frames": {},
    "frames_group_balance": {"group_balance": True},
    "frames_adv_session": {"adversarial": "session"},
    "frames_group_balance_adv_session": {"group_balance": True, "adversarial": "session"},
}
# Earlier results on the same folds and validation windows, for the summary table.
REFERENCE_ROWS = [
    {"run": "ref: feature baseline (HGB)", "steady_F1": 0.476, "50Hz": 0.539, "60Hz": 0.265,
     "untrimmed_1.4s": 0.458},
    {"run": "ref: raw-signal TCN", "steady_F1": 0.413, "50Hz": 0.471, "60Hz": 0.241,
     "untrimmed_1.4s": 0.442},
]

# (8, EMG_DIM) indicator matrices: which frame-feature columns belong to channel c.
_BAND_OF = np.zeros((8, EMG_DIM), np.float32)
_CORR_OF = np.zeros((8, EMG_DIM), bool)
for _c in range(8):
    _BAND_OF[_c, BAND_COLS_OF_CHANNEL[_c]] = 1.0
    _CORR_OF[_c, CORR_COLS_OF_CHANNEL[_c]] = True


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


# --------------------------------------------------------------------------- data

def _prep(job):
    w, kind = job
    e, i = preprocess(w.emg, w.imu)
    if kind == "frames":
        e, i = window_frames(e, i)
    return e, i, LABEL_TO_ID[w.label], w.session


class Pool:
    """All windows of one kind, padded once and kept on the device."""

    def __init__(self, windows, kind: str, device: torch.device, workers: int):
        with ProcessPoolExecutor(workers) as ex:
            items = list(ex.map(_prep, [(w, kind) for w in windows], chunksize=128))
        n, T = len(items), max(len(it[0]) for it in items)
        ce, ci = items[0][0].shape[1], items[0][1].shape[1]
        emg = np.zeros((n, T, ce), np.float32)
        imu = np.zeros((n, T, ci), np.float32)
        mask = np.zeros((n, T), bool)
        for k, (e, i, _, _) in enumerate(items):
            emg[k, :len(e)], imu[k, :len(i)], mask[k, :len(e)] = e, i, True
        self.emg = torch.from_numpy(emg).to(device)
        self.imu = torch.from_numpy(imu).to(device)
        self.mask = torch.from_numpy(mask).to(device)
        self.y = torch.tensor([it[2] for it in items], dtype=torch.long, device=device)
        self.session_np = np.array([it[3] for it in items])
        self.is60 = torch.from_numpy(np.isin(self.session_np, list(SESSIONS_60HZ))).to(device)
        self.y_np = self.y.cpu().numpy()

    def __len__(self):
        return len(self.y)

    def batch(self, idx: torch.Tensor):
        m = self.mask[idx]
        L = int(m.sum(1).max())                       # trim padding to this batch's longest
        return self.emg[idx, :L], self.imu[idx, :L], m[:, :L]


class Standardizer:
    """Per-feature mean/std over valid steps of the training windows (frames
    only; the raw signal is already gain-normalised). Also keeps the per-feature
    minimum, used to simulate a dead channel."""

    def __init__(self, pool: Pool, idx: torch.Tensor, enabled: bool):
        self.enabled = enabled
        if not enabled:
            return
        m = pool.mask[idx]
        e, i = pool.emg[idx][m], pool.imu[idx][m]
        self.e_mu, self.e_sd = e.mean(0), e.std(0).clamp(min=1e-6)
        self.i_mu, self.i_sd = i.mean(0), i.std(0).clamp(min=1e-6)
        self.e_min = e.min(0).values

    def __call__(self, e, i, m):
        if self.enabled:
            e = (e - self.e_mu) / self.e_sd
            i = (i - self.i_mu) / self.i_sd
        mf = m.unsqueeze(-1).float()
        return e * mf, i * mf

    def state(self) -> dict:
        if not self.enabled:
            return {}
        return {k: getattr(self, k).cpu() for k in ("e_mu", "e_sd", "i_mu", "i_sd", "e_min")}


def augment(e, i, kind: str, std: Standardizer, gen: torch.Generator):
    """Training-time augmentation on the device, in the input's own units."""
    B, dev = e.shape[0], e.device
    rnd = lambda *shape: torch.randn(*shape, generator=gen, device=dev)  # noqa: E731
    uni = lambda *shape: torch.rand(*shape, generator=gen, device=dev)   # noqa: E731
    drop = uni(B) < 0.15                                    # one bad contact in 15% of windows
    dead = torch.randint(0, 8, (B,), generator=gen, device=dev)
    if kind == "raw":
        e = e * torch.exp(0.15 * rnd(B, 1, 8))              # per-channel contact gain
        e = e + 0.05 * rnd(*e.shape)                        # sensor noise
        keep = torch.ones(B, 1, 8, device=dev)
        keep[drop, 0, dead[drop]] = 0.0
        e = e * keep
        i = i * torch.exp(0.1 * rnd(B, 1, 6)) + 0.05 * rnd(*i.shape)
        return e, i
    # frames: a channel gain g adds 2*ln(g) to that channel's log band powers;
    # correlations are gain-invariant. A dead channel gets the lowest band power
    # seen in training and zero correlation with every other channel.
    band_of = torch.from_numpy(_BAND_OF).to(dev)
    corr_of = torch.from_numpy(_CORR_OF).to(dev)
    e = e + (2 * 0.15 * rnd(B, 8) @ band_of).unsqueeze(1)
    dead_band = ((band_of[dead] > 0) & drop[:, None]).unsqueeze(1)      # (B, 1, D)
    dead_corr = (corr_of[dead] & drop[:, None]).unsqueeze(1)
    e = torch.where(dead_band, std.e_min.view(1, 1, -1).expand_as(e), e)
    e = e.masked_fill(dead_corr, 0.0)
    return e, i


def feature_noise(e, i, kind: str, gen: torch.Generator):
    """Noise in standardised units (frames only)."""
    if kind == "frames":
        e = e + 0.1 * torch.randn(e.shape, generator=gen, device=e.device)
        i = i + 0.1 * torch.randn(i.shape, generator=gen, device=i.device)
    return e, i


# ----------------------------------------------------------------------- training

def model_config(args) -> dict:
    cfg = dict(FRAME_CONFIG if args.input == "frames" else RAW_CONFIG)
    cfg.update(dropout=args.dropout, use_imu=not args.no_imu)
    return cfg


def train_model(pool: Pool, idx: torch.Tensor, args, device, seed: int):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    std = Standardizer(pool, idx, enabled=args.input == "frames")

    # domain labels for adversarial training
    if args.adversarial == "session":
        idx_np = idx.cpu().numpy()
        uniq = np.unique(pool.session_np[idx_np])
        dom = np.full(len(pool), -1, np.int64)
        dom[idx_np] = np.searchsorted(uniq, pool.session_np[idx_np])
        dom_all, n_domains = torch.from_numpy(dom).to(device), len(uniq)
    elif args.adversarial == "setup":
        dom_all, n_domains = pool.is60.long(), 2
    else:
        dom_all, n_domains = None, 0

    # loss weights: 50 Hz and 60 Hz sessions contribute equally
    w_all = torch.ones(len(pool), device=device)
    if args.group_balance:
        frac60 = pool.is60[idx].float().mean().clamp(1e-6, 1 - 1e-6)
        w_all = torch.where(pool.is60, 0.5 / frac60, 0.5 / (1 - frac60))

    model = AuraTCN(**model_config(args), n_domains=n_domains).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_batches = math.ceil(len(idx) / args.batch)
    total = args.epochs * n_batches
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.15)
    step = 0
    for epoch in range(args.epochs):
        model.train()
        perm = idx[torch.randperm(len(idx), generator=gen, device=device)]
        tot_loss = torch.zeros((), device=device)    # accumulate on the device: no per-step sync
        tot_dom = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for b in range(n_batches):
            bi = perm[b * args.batch:(b + 1) * args.batch]
            e, i, m = pool.batch(bi)
            e, i = augment(e, i, args.input, std, gen)
            e, i = std(e, i, m)
            e, i = feature_noise(e, i, args.input, gen)
            y, w = pool.y[bi], w_all[bi]
            if n_domains:
                p = step / max(1, total - 1)
                lam = args.adv_weight * (2 / (1 + math.exp(-10 * p)) - 1)   # DANN ramp-up
                logits, dlogits = model.forward_with_domain(e, i, m, lam)
                dloss = F.cross_entropy(dlogits, dom_all[bi])
            else:
                logits, dloss = model(e, i, m), torch.zeros((), device=device)
            ce = F.cross_entropy(logits, y, label_smoothing=0.05, reduction="none")
            loss = (ce * w).sum() / w.sum() + dloss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tot_loss += loss.detach() * len(bi)
            tot_dom += dloss.detach() * len(bi)
            correct += (logits.argmax(1) == y).sum()
        if args.verbose and (epoch % 5 == 4 or epoch == args.epochs - 1):
            n = len(idx)
            msg = (f"    epoch {epoch + 1:3d}: loss {float(tot_loss) / n:.3f} "
                   f"train acc {float(correct) / n:.3f}")
            if n_domains:
                msg += f" domain loss {float(tot_dom) / n:.3f}"
            print(msg, flush=True)
    return model, std


@torch.no_grad()
def predict(model, std: Standardizer, pool: Pool, idx: torch.Tensor, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(idx), batch):
        bi = idx[s:s + batch]
        e, i, m = pool.batch(bi)
        e, i = std(e, i, m)
        out.append(torch.softmax(model(e, i, m), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)), np.float32)


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", type=Path, nargs="?", default=Path("data/train_segments.npz"),
                    help="segment file (default data/train_segments.npz)")
    ap.add_argument("--sweep", action="store_true",
                    help="run all configurations in SWEEP on the same data, print a comparison")
    ap.add_argument("--input", choices=["frames", "raw"], default="frames")
    ap.add_argument("--per-session", type=int, default=600, help="training windows per session")
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--max-folds", type=int, default=None, help="run only the first N folds")
    ap.add_argument("--fold-ids", type=int, nargs="+", default=None,
                    help="run only these fold indices (0-based), e.g. to cover both setups")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--group-balance", action="store_true",
                    help="weight the loss so 50 Hz and 60 Hz sessions count equally")
    ap.add_argument("--adversarial", choices=["none", "setup", "session"], default="none")
    ap.add_argument("--adv-weight", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="mps", help="mps (default), cuda or cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print training loss every 5 epochs")
    ap.add_argument("--final", action="store_true", help="train on all sessions and save")
    ap.add_argument("--out", type=Path, default=Path("models/aura_tcn.pt"))
    ap.add_argument("--report", type=Path, default=Path("outputs/tcn_cv.json"))
    args = ap.parse_args()
    if args.sweep and args.final:
        ap.error("--sweep and --final cannot be combined")

    t0 = time.time()
    device = pick_device(args.device)
    segs = load_segments(args.segments)
    tr_w, va_w, raw_w = [], [], []
    for s, g in segs.groupby("session"):
        tr_w += WindowSampler(g, seed=args.seed * 1000 + s).sample_many(args.per_session)
        va_w += fixed_windows(g, length=1.2, stride=0.5)
        raw_w += fixed_windows(g, length=1.4, stride=0.5, trims=NO_TRIM)
    train = Pool(tr_w, args.input, device, args.workers)
    val = Pool(va_w, args.input, device, args.workers)
    raw = Pool(raw_w, args.input, device, args.workers)
    probe = AuraTCN(**model_config(args))
    print(f"device {device}; input {args.input}; {len(train)} train / {len(val)} steady val / "
          f"{len(raw)} untrimmed windows; model {count_parameters(probe):,} parameters "
          f"(receptive field {probe.receptive_field} {'frames' if args.input == 'frames' else 'samples'}); "
          f"group balance {args.group_balance}; adversarial {args.adversarial}; "
          f"data ready in {time.time() - t0:.0f} s", flush=True)

    if args.final:
        model, std = train_model(train, torch.arange(len(train), device=device), args, device, args.seed)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()
                 if not k.startswith("domain_head")}          # loadable without MPS
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state, "labels": LABELS, "input": args.input,
                    "config": model_config(args), "standardizer": std.state()}, args.out)
        print(f"saved {args.out} ({time.time() - t0:.0f} s)")
        return

    if args.sweep:
        rows = []
        for name, overrides in SWEEP.items():
            run_args = argparse.Namespace(**{**vars(args), **overrides})
            run_args.report = args.report.parent / f"tcn_{name}.json"
            print(f"\n=== {name}: group balance {run_args.group_balance}, "
                  f"adversarial {run_args.adversarial}", flush=True)
            t1 = time.time()
            report = cross_validate(train, val, raw, run_args, device, t0, show_details=False)
            rows.append({"run": name, "steady_F1": report["steady_macro_F1"],
                         "50Hz": report["50Hz_sessions"], "60Hz": report["60Hz_sessions"],
                         "untrimmed_1.4s": report["untrimmed_1.4s_macro_F1"],
                         "minutes": (time.time() - t1) / 60})
        table = pd.DataFrame(rows + REFERENCE_ROWS).set_index("run")
        print("\nSummary: macro F1, sessions held out (same folds as the baseline)")
        print(table.round(3).to_string())
        out = args.report.parent / "tcn_sweep.csv"
        table.to_csv(out)
        print(f"saved {out} plus one report and probabilities file per run; total "
              f"{(time.time() - t0) / 60:.1f} min")
        return

    cross_validate(train, val, raw, args, device, t0, show_details=True)


def cross_validate(train: Pool, val: Pool, raw: Pool, args, device, t0: float,
                   show_details: bool = True) -> dict:
    """Session-held-out CV for one configuration; writes the report and
    probabilities next to ``args.report`` and returns the report."""
    probe = AuraTCN(**model_config(args))
    sessions = np.unique(train.session_np)
    folds = session_folds(sessions, args.folds)                # same folds as scripts/baseline.py
    prob_val = np.full((len(val), len(LABELS)), np.nan, np.float32)
    prob_raw = np.full((len(raw), len(LABELS)), np.nan, np.float32)
    run = args.fold_ids if args.fold_ids is not None else range(len(folds))[:args.max_folds]
    for k in run:
        held = folds[k].tolist()
        # index sets built in NumPy (torch.isin is not available on every MPS build)
        tr_idx = torch.from_numpy(np.flatnonzero(~np.isin(train.session_np, held))).to(device)
        model, std = train_model(train, tr_idx, args, device, args.seed + k)
        for pool, prob in ((val, prob_val), (raw, prob_raw)):
            idx_np = np.flatnonzero(np.isin(pool.session_np, held))
            prob[idx_np] = predict(model, std, pool, torch.from_numpy(idx_np).to(device))
        vi = np.isin(val.session_np, held)
        f1 = macro_f1(val.y_np[vi], prob_val[vi].argmax(1))
        print(f"fold {k + 1}/{args.folds} sessions {held}: steady macro F1 {f1:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    done = ~np.isnan(prob_val[:, 0])
    y, p, sess = val.y_np[done], prob_val[done].argmax(1), val.session_np[done]
    g60 = np.isin(sess, list(SESSIONS_60HZ))
    rdone = ~np.isnan(prob_raw[:, 0])
    report = {
        "steady_macro_F1": macro_f1(y, p),
        "50Hz_sessions": macro_f1(y[~g60], p[~g60]) if (~g60).any() else None,
        "60Hz_sessions": macro_f1(y[g60], p[g60]) if g60.any() else None,
        "untrimmed_1.4s_macro_F1": macro_f1(raw.y_np[rdone], prob_raw[rdone].argmax(1)),
        "per_class_F1": dict(zip(LABELS, f1_per_class(y, p, range(6)).round(3).tolist())),
        "per_session_F1": {int(s): round(macro_f1(y[sess == s], p[sess == s]), 3)
                           for s in np.unique(sess)},
        "n_parameters": count_parameters(probe),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    summary = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report.items()
               if k not in ("args", "per_session_F1") or show_details}
    summary.pop("args", None)
    print(json.dumps(summary, indent=1))
    if show_details:
        print("Confusion matrix (rows = true):")
        print(pd.DataFrame(confusion(y, p, 6), index=LABELS, columns=LABELS))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))
    probs_path = args.report.with_suffix(".probs.npz")
    np.savez_compressed(probs_path, prob_val=prob_val, y_val=val.y_np, session_val=val.session_np,
                        prob_raw=prob_raw, y_raw=raw.y_np, session_raw=raw.session_np, labels=LABELS)
    print(f"saved {args.report} and {probs_path}", flush=True)
    return report


if __name__ == "__main__":
    main()
