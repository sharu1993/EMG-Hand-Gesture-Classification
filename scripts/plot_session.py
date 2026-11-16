"""Plot one Aura training session: filtered EMG, accelerometer and gyroscope,
with the guided gesture labels drawn as boxes on a label track.

Usage
-----
    python scripts/plot_session.py vr-calibration-free-emg-challenge-pieeg/train/train_1.xlsx
    python scripts/plot_session.py train_1.xlsx --start 20 --end 50 --out plots/train_1_20-50.png
    python scripts/plot_session.py train_1.xlsx --raw          # skip EMG filtering

EMG is high-pass filtered (20 Hz) and notch filtered at the mains frequency and
its harmonics (50 or 60 Hz, detected per session), then each channel is scaled
by its robust std (MAD / 0.6745) so all eight are comparable on one axis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

EMG_CH = [f"ch{i}" for i in range(1, 9)]
ACC = ["ax", "ay", "az"]
GYR = ["gx", "gy", "gz"]

# Reference categorical palette (light mode). Gestures take slots 4-8 so that
# the IMU x/y/z lines (slots 1-3) never share a hue with a label.
GESTURE_COLORS = {
    "fist": "#eda100",      # yellow
    "open": "#e87ba4",      # magenta
    "pinch": "#008300",     # green
    "point": "#6250d6",     # violet
    "thumb-up": "#e34948",  # red
}
REST_COLOR = "#e1e0d9"      # neutral gray: rest is a baseline, not a series
XYZ_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"


def load_session(path: Path) -> pd.DataFrame:
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_excel(path, sheet_name=0, engine="openpyxl")


def effective_fs(t: np.ndarray) -> float:
    """Rows per second. Timestamps are packet arrival times (many duplicates),
    so the sample index, not t, is the reliable time base for filtering."""
    return (len(t) - 1) / (t[-1] - t[0])


def detect_mains(x: np.ndarray, fs: float) -> int:
    f, p = welch(x, fs=fs, nperseg=1024, axis=0)
    p = p.mean(axis=1)
    band = lambda lo, hi: p[(f >= lo) & (f < hi)].mean()
    return 50 if band(49, 52) > band(59, 62) else 60


def filter_emg(x: np.ndarray, fs: float) -> tuple[np.ndarray, int]:
    x = x - np.median(x, axis=0)
    y = sosfiltfilt(butter(4, 20, btype="high", fs=fs, output="sos"), x, axis=0)
    mains = detect_mains(y, fs)
    for f0 in range(mains, int(fs / 2), mains):
        y = sosfiltfilt(tf2sos(*iirnotch(f0, Q=30, fs=fs)), y, axis=0)
    return y, mains


def label_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of guided labels -> (label, t_start, t_end)."""
    lab = df["label"].where(df["label_source"] == "guided")
    run = (lab != lab.shift()).cumsum()
    segs = (
        df.assign(lab=lab, run=run)
        .dropna(subset=["lab"])
        .groupby("run")
        .agg(label=("lab", "first"), t0=("t", "first"), t1=("t", "last"))
    )
    return segs.reset_index(drop=True)


def style_axis(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")


def plot_session(df: pd.DataFrame, title: str, start: float | None, end: float | None,
                 raw: bool = False, env_ms: int = 200) -> plt.Figure:
    t = df["t"].to_numpy()
    fs = effective_fs(t)
    emg = df[EMG_CH].to_numpy(dtype=float)
    if raw:
        emg, mains_note = emg - np.median(emg, axis=0), "raw, median-removed"
    else:
        emg, mains = filter_emg(emg, fs)
        mains_note = f"HP 20 Hz + {mains} Hz notches"
    scale = np.median(np.abs(emg), axis=0) / 0.6745 + 1e-9

    lo = t[0] if start is None else start
    hi = t[-1] if end is None else end
    m = (t >= lo) & (t <= hi)
    imu = m & (df["imu_new"].to_numpy() == 1)  # IMU values are held between updates
    segs = label_segments(df)
    segs = segs[(segs.t1 >= lo) & (segs.t0 <= hi)]

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 11), sharex=True,
        gridspec_kw={"height_ratios": [0.5, 6, 1.6, 1.6], "hspace": 0.08},
    )
    fig.patch.set_facecolor(SURFACE)
    ax_lab, ax_emg, ax_acc, ax_gyr = axes

    # Label track: one box per guided segment, named inside when it fits.
    ax_lab.set_facecolor(SURFACE)
    ax_lab.set_ylim(0, 1)
    ax_lab.set_yticks([])
    ax_lab.tick_params(axis="x", length=0)
    for side in ax_lab.spines.values():
        side.set_visible(False)
    width_s = hi - lo
    for s in segs.itertuples():
        color = GESTURE_COLORS.get(s.label, REST_COLOR)
        ax_lab.add_patch(Rectangle((s.t0, 0.1), s.t1 - s.t0, 0.8, facecolor=color,
                                   edgecolor=SURFACE, linewidth=1))
        if (s.t1 - s.t0) / width_s > 0.03 and s.label != "rest":  # only if the name fits
            txt = INK if s.label == "fist" else "#ffffff"  # dark text on yellow
            ax_lab.text((s.t0 + s.t1) / 2, 0.5, s.label, ha="center", va="center",
                        fontsize=8, color=txt, clip_on=True)
    ax_lab.set_title(title, loc="left", color=INK, fontsize=12, pad=24)

    # Gesture spans shaded in neutral gray on the data panels (identity lives on
    # the label track, so the data lines keep their own colors).
    for ax in (ax_emg, ax_acc, ax_gyr):
        for s in segs[segs.label != "rest"].itertuples():
            ax.axvspan(s.t0, s.t1, color="#f0efec", zorder=0, linewidth=0)

    # EMG: eight channels stacked on one axis, each scaled by its robust std.
    # Light trace = filtered signal; dark line = moving RMS envelope (activity level).
    gap = 8.0
    win = max(1, int(round(env_ms / 1000 * fs)))
    kernel = np.ones(win) / win
    for i, ch in enumerate(EMG_CH):
        y = emg[:, i] / scale[i]
        rms = np.sqrt(np.convolve(y ** 2, kernel, mode="same"))
        ax_emg.plot(t[m], np.clip(y[m], -gap / 2, gap / 2) - i * gap,
                    color="#c3c2b7", linewidth=0.4, rasterized=True)
        ax_emg.plot(t[m], np.clip(2 * rms[m], 0, gap / 2) - i * gap,
                    color=INK, linewidth=0.9, rasterized=True)
    ax_emg.set_yticks([-i * gap for i in range(8)], EMG_CH)
    ax_emg.set_ylim(-7.6 * gap, 0.6 * gap)
    style_axis(ax_emg, f"EMG ({mains_note}; per-channel σ units)\n"
                       f"gray: signal  ·  black: 2×RMS envelope, {env_ms} ms")

    for ax, cols, unit in ((ax_acc, ACC, "Accel (g)"), (ax_gyr, GYR, "Gyro (°/s)")):
        for c, color in zip(cols, XYZ_COLORS):
            ax.plot(t[imu], df.loc[imu, c], color=color, linewidth=1.2, label=c[-1])
        style_axis(ax, unit)
        ax.legend(loc="lower right", bbox_to_anchor=(1, 0.98), ncol=3, frameon=False,
                  fontsize=8, labelcolor=INK_2, handlelength=1.5, borderaxespad=0)
    ax_gyr.set_xlabel("time (s)", color=INK_2, fontsize=9)
    ax_gyr.set_xlim(lo, hi)

    handles = [Patch(facecolor=c, label=k) for k, c in GESTURE_COLORS.items()]
    handles.append(Patch(facecolor=REST_COLOR, label="rest"))
    fig.legend(handles=handles, loc="upper right", ncol=6, frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.9, 0.93))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, help="train_N.xlsx (or a cached .pkl)")
    ap.add_argument("--start", type=float, help="window start (s)")
    ap.add_argument("--end", type=float, help="window end (s)")
    ap.add_argument("--raw", action="store_true", help="plot EMG without filtering")
    ap.add_argument("--env-ms", type=int, default=200, help="RMS envelope window (ms)")
    ap.add_argument("--out", type=Path, help="output PNG (default plots/<name>.png)")
    args = ap.parse_args()

    df = load_session(args.session)
    name = args.session.stem.replace("train_train_", "train_")
    window = "" if args.start is None and args.end is None else \
        f"  [{args.start or 0:g}-{args.end if args.end is not None else 'end'} s]"
    fig = plot_session(df, f"{name}{window}", args.start, args.end, raw=args.raw,
                       env_ms=args.env_ms)
    out = args.out or Path("plots") / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
