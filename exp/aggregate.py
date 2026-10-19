# -*- coding: utf-8 -*-
"""汇总结果为论文表 1/2/3 格式（均值±标准差，5 种子）。

用法: python exp/aggregate.py [--filter tag=main] [--metrics acc,f1,auc]
"""
import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
CORE_METRICS = ["acc", "precision", "recall", "f1", "auc", "fpr", "fnr",
                "rmse_delta_deg", "rmse_omega_pu", "latency_ms", "throughput_sps"]


def collect():
    rows = []
    for f in sorted(RESULTS.glob("*/*.json")):
        with open(f, encoding="utf-8") as fp:
            rows.append(json.load(fp))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="", help="如 tag=main 或 model=pimtnet,dataset=ieee39")
    ap.add_argument("--metrics", default="acc,f1,auc,fnr")
    args = ap.parse_args()
    flt = dict(kv.split("=") for kv in args.filter.split(",") if kv)
    rows = collect()
    for k, v in flt.items():
        rows = [r for r in rows if str(r.get(k)) == v]
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["dataset"], r["tag"]), []).append(r)

    mets = args.metrics.split(",")
    print(f"{'model':<12}{'dataset':<9}{'tag':<8}{'n':>2}  " +
          "  ".join(f"{m:>18}" for m in mets))
    for (model, dataset, tag), rs in sorted(groups.items()):
        cells = []
        for m in mets:
            vals = [r[m] for r in rs if r.get(m) is not None and np.isfinite(r[m])]
            cells.append(f"{np.mean(vals):.4f}±{np.std(vals):.4f}" if vals else "-")
        print(f"{model:<12}{dataset:<9}{tag:<8}{len(rs):>2}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
