"""Split Aura training sessions into labelled gesture segments.

A segment is a contiguous run of rows that share one guided label
(``label_source == "guided"``). Rows labelled by the optical tracker or not at
all (transitions) are dropped. Each segment becomes one sample:

    sample_id   "train07_s012": session number + segment index within the session
    session     source session number (7 for train_7.xlsx)
    seg_idx     segment index within the session, in time order
    label       rest, fist, open, pinch, point, thumb-up
    t_start     first timestamp of the segment (s, session clock)
    t_end       last timestamp of the segment (s, session clock)
    duration    t_end - t_start (s)
    n_rows      number of rows (samples) in the segment
    t           (n_rows,) timestamps, session clock
    emg         (n_rows, 8) raw EMG ch1..ch8, microvolts, no filtering
    imu         (n_rows, 6) ax, ay, az (g), gx, gy, gz (deg/s)
    imu_new     (n_rows,) 1 where the IMU produced a fresh reading, 0 where held
    orientation (n_rows, 3) roll, pitch, yaw (deg); train only, not in the test set

Signals are stored raw so that filtering, windowing and normalisation stay
decisions of the modelling code.

Output is a NumPy archive (portable across pandas versions) plus a CSV index of
the metadata for quick inspection. Load it back with ``load_segments``.

Usage
-----
    python -m emg_gestures.segments vr-calibration-free-emg-challenge-pieeg/train \
        --out data/train_segments.npz

    from emg_gestures.segments import load_segments
    segs = load_segments("data/train_segments.npz")   # one row per segment
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

EMG_COLS = [f"ch{i}" for i in range(1, 9)]
IMU_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
ORI_COLS = ["roll", "pitch", "yaw"]
LABELS = ["rest", "fist", "open", "pinch", "point", "thumb-up"]


def session_number(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    if not m:
        raise ValueError(f"cannot read a session number from {path.name}")
    return int(m.group(1))


def load_session(path: Path) -> pd.DataFrame:
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_excel(path, sheet_name=0, engine="openpyxl")


def parse_session(df: pd.DataFrame, session: int) -> list[dict]:
    """Return one dict per contiguous guided-label segment, in time order."""
    label = df["label"].where(df["label_source"] == "guided")
    run_id = (label != label.shift()).cumsum().to_numpy()
    keep = label.notna().to_numpy()

    samples = []
    for seg_idx, rid in enumerate(pd.unique(run_id[keep])):
        rows = np.flatnonzero(run_id == rid)
        seg = df.iloc[rows]
        t = seg["t"].to_numpy(dtype=float)
        samples.append({
            "sample_id": f"train{session:02d}_s{seg_idx:03d}",
            "session": session,
            "seg_idx": seg_idx,
            "label": seg["label"].iat[0],
            "t_start": t[0],
            "t_end": t[-1],
            "duration": t[-1] - t[0],
            "n_rows": len(rows),
            "t": t,
            "emg": seg[EMG_COLS].to_numpy(dtype=np.float64),
            "imu": seg[IMU_COLS].to_numpy(dtype=np.float64),
            "imu_new": seg["imu_new"].to_numpy(dtype=np.int8),
            "orientation": seg[ORI_COLS].to_numpy(dtype=np.float64),
        })
    return samples


def _parse_file(path: Path) -> list[dict]:
    return parse_session(load_session(path), session_number(path))


def build_segments(train_dir: Path, workers: int = 2) -> pd.DataFrame:
    files = sorted(
        [p for p in train_dir.iterdir() if p.suffix in (".xlsx", ".pkl") and not p.name.startswith("~$")],
        key=session_number,
    )
    if not files:
        raise FileNotFoundError(f"no train_*.xlsx files in {train_dir}")
    with ProcessPoolExecutor(workers) as ex:
        per_file = list(ex.map(_parse_file, files))
    df = pd.DataFrame([s for samples in per_file for s in samples])
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"unexpected labels: {sorted(unknown)}")
    return df


META_COLS = ["sample_id", "session", "seg_idx", "label", "t_start", "t_end", "duration", "n_rows"]
ARRAY_COLS = ["t", "emg", "imu", "imu_new", "orientation"]


def save_segments(df: pd.DataFrame, path: Path) -> None:
    """Concatenate the per-segment arrays and store them with row offsets."""
    offsets = np.r_[0, np.cumsum(df["n_rows"].to_numpy())]
    arrays = {c: np.concatenate(df[c].to_list()) for c in ARRAY_COLS}
    meta = {f"meta_{c}": df[c].to_numpy(dtype=str if c in ("sample_id", "label") else None)
            for c in META_COLS}
    np.savez_compressed(path, offsets=offsets, **arrays, **meta)


def load_segments(path: str | Path) -> pd.DataFrame:
    """Inverse of save_segments: one row per segment, arrays as per-row columns."""
    z = np.load(path, allow_pickle=False)
    df = pd.DataFrame({c: z[f"meta_{c}"] for c in META_COLS})
    off = z["offsets"]
    for c in ARRAY_COLS:
        a = z[c]
        df[c] = [a[off[i]:off[i + 1]] for i in range(len(df))]
    return df


def summarise(df: pd.DataFrame) -> str:
    counts = df.pivot_table(index="label", values="duration",
                            aggfunc=["count", "mean", "min", "max"]).round(2)
    counts.columns = ["n_segments", "mean_s", "min_s", "max_s"]
    return (f"{len(df)} segments from {df['session'].nunique()} sessions\n"
            f"{counts.reindex(LABELS).to_string()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_dir", type=Path, help="folder with train_*.xlsx")
    ap.add_argument("--out", type=Path, default=Path("data/train_segments.npz"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    df = build_segments(args.train_dir, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_segments(df, args.out)
    df[META_COLS].round(3).to_csv(args.out.with_suffix(".csv"), index=False)
    print(summarise(df))
    print(f"saved {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
