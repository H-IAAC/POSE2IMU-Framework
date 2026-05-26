"""
Export each subplot from 'All sensors — accelerometer + gyroscope' individually
as vector PDF files (no rasterisation).
Output: output/imu_subplots_pdf/<sensor>_<acc|gyr>.pdf
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("pdf")          # vector backend — no rasterisation
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
SENSOR_NAMES = ["waist", "head", "right_forearm", "left_forearm"]
CHANNELS     = ["ax", "ay", "az", "gx", "gy", "gz"]
ACC_IDX      = [0, 1, 2]
GYR_IDX      = [3, 4, 5]

ROOT       = Path("../output/robot_emotions_virtual_imu")
OUT_DIR    = Path("../output/imu_subplots_pdf")
FILE_IDX   = 0          # which clip to use (same as notebook default)

COLORS_ACC = ["#e74c3c", "#2ecc71", "#3498db"]
COLORS_GYR = ["#c0392b", "#27ae60", "#2980b9"]

# ── Load data ─────────────────────────────────────────────────────────────────
all_files = sorted(ROOT.rglob("imu.npz"))
if not all_files:
    sys.exit(f"No imu.npz files found under {ROOT}")

path = all_files[FILE_IDX]
data = np.load(path, allow_pickle=True)
t    = data["timestamps_sec"]
imu  = data["imu"]               # (T, 4, 6)
t    = t - t[0]

clip_name = path.parent.name
print(f"Clip : {clip_name}  |  shape: {imu.shape}  |  {t[-1]:.1f} s")

# ── Export ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

for s, sensor in enumerate(SENSOR_NAMES):
    # — accelerometer subplot —
    fig, ax = plt.subplots(figsize=(8, 3))
    for i, idx in enumerate(ACC_IDX):
        ax.plot(t, imu[:, s, idx],
                color=COLORS_ACC[i], lw=0.8, alpha=0.9, label=CHANNELS[idx])
    ax.set_ylabel("m/s²")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{sensor} — accelerometer  |  {clip_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / f"{sensor}_accelerometer.pdf"
    fig.savefig(out_path, format="pdf", backend="pdf")
    plt.close(fig)
    print(f"  saved {out_path}")

    # — gyroscope subplot —
    fig, ax = plt.subplots(figsize=(8, 3))
    for i, idx in enumerate(GYR_IDX):
        ax.plot(t, imu[:, s, idx],
                color=COLORS_GYR[i], lw=0.8, alpha=0.9, label=CHANNELS[idx])
    ax.set_ylabel("rad/s")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{sensor} — gyroscope  |  {clip_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / f"{sensor}_gyroscope.pdf"
    fig.savefig(out_path, format="pdf", backend="pdf")
    plt.close(fig)
    print(f"  saved {out_path}")

print(f"\nDone. {len(SENSOR_NAMES) * 2} PDFs written to {OUT_DIR.resolve()}")
