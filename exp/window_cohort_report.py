# -*- coding: utf-8 -*-
"""固定队列（1,642 场景）窗口消融汇总 -> results/window_cohort_summary.json。

队列曲线：t=0.2/0.5/1.0/2.0s 用 *_wc 结果（results/<model>_wc/ieee39c*_main_seed*.json），
4.0s 点复用 w241 现有 main 结果（w241 存储 split == 队列 split，test 即队列 328 场景）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402

ms = mf.ms
CWD = [("ieee39c13", 0.2), ("ieee39c31", 0.5), ("ieee39c61", 1.0),
       ("ieee39c121", 2.0)]

out = {}
for m in ("pimtnet", "svm", "xgboost"):
    row = {}
    for ds, t in CWD:
        mu, sd, n = ms(f"{m}_wc", ds)
        row[str(t)] = dict(acc=round(100 * mu, 2), std=round(100 * sd, 2), n=n)
    mu, sd, n = ms(m, "ieee39w241")   # 4.0s：队列训练划分与 w241 完全一致
    row["4.0"] = dict(acc=round(100 * mu, 2), std=round(100 * sd, 2), n=n)
    out[m] = row

dst = Path(__file__).resolve().parent / "results" / "window_cohort_summary.json"
dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=2))
