# -*- coding: utf-8 -*-
"""P1-9 验证：0.2s 窗内 fault-on 场景比例（dur > 0.2s）。"""
import numpy as np
from pathlib import Path

f = Path(__file__).resolve().parent.parent / "data" / "processed" / "ieee39_windows.npz"
d = np.load(f)
dur = d["desc"][:, 3]
n = len(dur)
print(f"n={n}, dur range=[{dur.min():.3f},{dur.max():.3f}]s")
for thr in (0.2,):
    frac = (dur > thr).mean()
    print(f"dur > {thr}s (fault-on at window end): {int((dur>thr).sum())} = {frac*100:.1f}%")
print("dur quantiles:", np.round(np.quantile(dur, [0, .25, .5, .75, 1]), 3))
# has_trip 维度
trip = d["desc"][:, 4]
print(f"has_trip rate: {trip.mean()*100:.1f}%")
