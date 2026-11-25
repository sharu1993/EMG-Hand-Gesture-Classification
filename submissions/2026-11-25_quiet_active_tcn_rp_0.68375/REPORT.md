# Submission: quiet/active trees + active TCNs + rest-vs-pinch TCNs

| | |
|---|---|
| Competition | PiEEG Kaggle "Aura: Calibration-Free EMG Challenge" |
| Submitted | 2026-11-25 |
| Public score (macro F1) | **0.68375** (previous best 0.65041) |
| Code | commit `c31c57d` on `work-branch` (copy in `code/`) |
| Run | `runs/quiet_active_tcn_rp` |
| Command | `python3 scripts/submit.py --name quiet_active_tcn_rp --tcn-rest-pinch 3` |

## What the model does

Each test action (1.4 s, 8 EMG channels + IMU) is classified on its own: no
calibration and no statistics shared across test files.

1. **Preprocessing** (`code/aura/preprocess.py`): high-pass at 20 Hz, then
   least-squares subtraction of 50/60/100/120 Hz sines, then band-stops at
   48–52, 58–62 and 118–122 Hz. Bands used: 20–45, 65–95 and 100–117 Hz.
2. **Features** (`code/aura/features.py`): absolute log band power (27);
   scale-free time course, spectral ratios and IMU onset (21); gain-normalised
   covariance shape/scale and correlation in log-Euclidean tangent space, plus IMU (202).
3. **Trees** (`code/aura/twostage.py`): three HistGradientBoosting models, averaged over 3 seeds.
   - Stage 1: quiet (rest + pinch) vs active.
   - Quiet branch: rest vs pinch, trained on equal numbers of rest and pinch windows.
   - Active branch: fist, open, point, thumb-up.
   - Gain augmentation for the models that involve rest.
4. **TCNs** on channel-relationship frames (`code/aura/sequence.py`): 0.4 s
   frames every 0.1 s, 131 features per frame.
   - 3 active-gesture TCNs (seeds 1000–1002): their mean is averaged half and half with the trees' active-branch probabilities.
   - 3 rest-vs-pinch TCNs (seeds 2000–2002), each trained on all pinch windows plus a different equal random set of rest windows: their mean p(pinch) is averaged half and half with the trees' quiet branch.
5. **Decision** (`code/aura/twostage.py`, `combine`): quiet if p(quiet) ≥ 0.5,
   then pinch if p(pinch | quiet) ≥ 0.5, else rest; otherwise the most likely
   active gesture. Fixed gates, no class weights or thresholds tuned on the test set.

Training windows: 20,000. Test-like windows start 0.5–1.0 s after the cue,
and steady windows are added (`code/aura/windows.py`).

## Validation

The scores come from cross-validation by session: 7 folds of 4 sessions, with
one 60 Hz session per fold (`code/config/folds.json`), repeated 3 times.
Each figure is six-class macro F1, the mean of the 3 repeats.

| Model | CV, gestures at 0.7 s | CV, gestures at 0.9 s | Public |
|---|---|---|---|
| Quiet/active trees | 0.656 | 0.638 | 0.573 |
| + active TCNs | 0.666 | 0.653 | 0.650 |
| **+ rest-vs-pinch TCNs (this submission)** | **0.669** | **0.660** | **0.684** |

- Per repeat, this model beat the trees alone by +0.002, +0.024 and +0.015.
- The rest-vs-pinch TCNs on top of the active TCNs scored worse than without them in one repeat of 3.
- Public-score noise is about ±0.05, so the public gain of +0.033 over the previous best is within noise.

Predicted classes on the 348 test actions: rest 186, thumb-up 39, open 39,
point 33, fist 27, pinch 24.

## Files

| File | What it is |
|---|---|
| `submission.csv` | the scored file (`action_id,gesture`) |
| `test_probabilities.csv` | per-action probabilities: trees and TCNs separately, and combined |
| `manifest.json` | the run's record: settings, checksums of inputs and outputs |
| `weights/models.pkl` | the 3 tree models (scikit-learn 1.8.0) |
| `weights/tcn_active.pt` | the 3 active-gesture TCNs: weights and input standardisation |
| `weights/tcn_rest_pinch.pt` | the 3 rest-vs-pinch TCNs: weights and input standardisation |
| `predict.py` | rebuilds the CSV from the saved weights, without training |
| `code/` | the code at commit `c31c57d`: `aura/`, the 4 pipeline scripts, `config/folds.json` |
| `environment.txt` | the Python packages on the training machine (Mac, Apple GPU) |

## Reproducing

**From the saved weights** (no training), from the repository root:

```
python3 submissions/2026-11-25_quiet_active_tcn_rp_0.68375/predict.py
```

This needs `data/features/test.pkl` and `data/features/frames_relations_test.npz`.
It writes `predicted.csv` here and reports how far it agrees with `submission.csv`.

**From scratch**, at commit `c31c57d`, with the Kaggle data in `vr-calibration-free-emg-challenge-pieeg/`:

```
python3 scripts/prepare_data.py
python3 scripts/build_features.py
python3 scripts/build_frames.py --kind relations
python3 scripts/submit.py --name quiet_active_tcn_rp --tcn-rest-pinch 3
```

The TCNs were trained on the Apple GPU (MPS), which is not guaranteed to be
bit-for-bit deterministic, so retraining may change a few labels. Whether it
does has not been checked.

## Checks done

- `scripts/submit.py` at `c31c57d` is byte-identical (SHA-256 `b9cd8e03…`) to the file that made this run.
  - The run started before that commit, so `manifest.json` records commit `577fb86` plus an uncommitted `scripts/submit.py`.
- The test features and relation frames used have the checksums recorded in `manifest.json`.
- Reloading the TCN weights reproduces the saved TCN probabilities to within 6×10⁻⁷ (on CPU).
- Combined with the saved tree probabilities, they give the same label as `submission.csv` for all 348 actions.
- The full rebuild from the saved weights on the training Mac (`predict.py`) matches `submission.csv` on all 348 actions.
- The tree weights can only be reloaded with scikit-learn 1.8.0, the version that saved them. Use the training machine's environment (`environment.txt`).

## Checksums (SHA-256)

```
28eec39c1efc06c89e6c1618a7818f62cd63602578070b1e4fb4bdd63f945098  submission.csv
a380336322e69e691092a9ca25a7715d0401e20dc0f69f83496d622bf88b86ce  weights/models.pkl
a0c4306b76957f1ee4de5b571b8c13e8502d811546a6209b7276f177b668d57d  weights/tcn_active.pt
fe80daad53ed3b24ff757518f050fc070e2388af30a7e0890f0138cf129cc73f  weights/tcn_rest_pinch.pt
8f4f2c4662b643e298656e71e72cbd7f39be11c64131f5816a10a886d5002cd5  data/features/train.pkl
504bf897af3416e0b9a847c486fbf70a4fb6c9ecbab83a44395b4036cc246f72  data/features/test.pkl
d647117f052eda565662a76ac1c986e2d7c53dfbce03cf1a491a85f0bb04413b  data/features/frames_relations_train.npz
63a494046ff9536636688f4d6882de9f100ec3d2fb2c0e0c9f30d2373573ac98  data/features/frames_relations_test.npz
```
