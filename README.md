# EMG Hand Gesture Classification

Entry for PiEEG's Kaggle competition
[Aura: Calibration-Free EMG Challenge](https://www.kaggle.com/competitions/vr-calibration-free-emg-challenge-pieeg):
classify hand gestures from 8-channel wrist EMG + IMU (Aura VR device) on people
not seen during training.

- **Classes:** rest, fist, open, pinch, point, thumb-up
- **Metric:** macro F1
- **Test input:** 1.4 s actions, each channel mean-centred per action

## Data

The competition data is not tracked in git. Download it from the Kaggle
competition page and place it in this folder:

```
vr-calibration-free-emg-challenge-pieeg/
├── train/train_1.xlsx … train_28.xlsx
├── test/test_1.xlsx … test_5.xlsx
└── sample_submission.csv
```
