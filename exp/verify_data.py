# -*- coding: utf-8 -*-
"""PLSA 论文数据集完整性验证脚本。

用法（在 plsa 环境下）:
    python exp/verify_data.py

验证对象（对应论文 4.1 节五个数据集中可获得的四个）:
  1. IEEE39-Sim  自建 2400 场景
  2. IEEE118-Sim 自建 1200 场景
  3. PF-39       公开 12852 次仿真（注意: 官方标签 1=stable / 0=unstable，与本文 1=失稳 相反）
  4. NE39-Open   公开特征版（当前仅 3120 行，论文声称 9360）
"""
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def derive_labels(res_vals):
    """按论文规则计算标签: status=diverged 或最大功角差>180° -> 1(失稳)。"""
    labels, spreads = [], []
    for r in res_vals:
        status, payload = r[1], r[2]
        if not isinstance(payload, tuple) or payload is None:
            labels.append(None)
            spreads.append(None)
            continue
        _, delta, _, _ = payload
        spread = float(np.rad2deg(delta.max(axis=1) - delta.min(axis=1)).max())
        labels.append(1 if (status == "diverged" or spread > 180.0) else 0)
        spreads.append(spread)
    return labels, spreads


def check_sim_pkl(path: Path, expect_n: int) -> bool:
    hr(f"[SIM] {path.name}  (expect {expect_n} scenarios)")
    if not path.exists():
        print(f"  MISSING: {path}")
        return False
    t0 = time.time()
    with open(path, "rb") as f:
        d = pickle.load(f)
    plans, results = d["plans"], d["results"]
    dt = time.time() - t0
    # results 为 dict {idx: (idx, status, (t, delta, omega, Pe))}
    plan_vals = list(plans.values()) if isinstance(plans, dict) else list(plans)
    res_vals = list(results.values()) if isinstance(results, dict) else list(results)
    ok = len(res_vals) == expect_n
    print(f"  container: plans {type(plans).__name__}, results {type(results).__name__}")
    print(f"  scenarios: {len(res_vals)} (expect {expect_n}) -> {'OK' if ok else 'MISMATCH'}")
    print(f"  plans[0] (case,cid,load,M,bus,tf,dur,trip_line): {plan_vals[0]}")
    status = Counter(r[1] for r in res_vals)
    print(f"  status dist: {dict(status)}")
    labels, spreads = derive_labels(res_vals)
    n_lab = [l for l in labels if l is not None]
    print(f"  label dist (1=unstable): {dict(Counter(n_lab))} "
          f"-> unstable {sum(n_lab)/max(len(n_lab),1)*100:.1f}%")
    ok_spread = [s for s, l in zip(spreads, labels) if l == 0 and s is not None]
    if ok_spread:
        print(f"  max-spread of stable cases: max {max(ok_spread):.1f} deg")
    # 波形结构抽查
    r0 = res_vals[0][2]
    t, delta = r0[0], r0[1]
    print(f"  waveform blocks: {[getattr(x, 'shape', type(x).__name__) for x in r0]}")
    print(f"  time span: {t[0]:.4f} ~ {t[-1]:.4f} s, dt~{np.diff(t).mean():.5f} s")
    print(f"  load time: {dt:.1f}s")
    return ok


def check_pf39() -> bool:
    hr("[PF-39] tsa_dataset  (expect 12852 sims, 60 steps x 50 feats)")
    ds_dir = DATA / "PF-39" / "tsa_dataset"
    if not (ds_dir / "data" / "tsa_data.pkl").exists():
        print(f"  MISSING: {ds_dir / 'data' / 'tsa_data.pkl'}")
        return False
    import os
    old_cwd = os.getcwd()
    os.chdir(ds_dir)  # 官方 loader 使用相对路径
    try:
        sys.path.insert(0, str(ds_dir))
        t0 = time.time()
        import dataset_loader  # noqa: E402
        bundle = dataset_loader.load_tsa_data()
        dt = time.time() - t0
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(ds_dir))
        sys.modules.pop("dataset_loader", None)
    data, target = bundle.data, bundle.target
    print(f"  data shape: {data.shape}, dtype {data.dtype}")
    print(f"  target dist (official: 1=stable, 0=unstable): {dict(Counter(target.tolist()))}")
    unstable_pct = (target == 0).mean() * 100
    print(f"  unstable share: {unstable_pct:.1f}% (paper expects ~42%)")
    print(f"  n features: {len(bundle.feature_names)}, e.g. {bundle.feature_names[:5]}")
    print(f"  load time: {dt:.1f}s")
    return data.shape[0] == 12852


def check_ne39() -> bool:
    hr("[NE39-Open] PowerGridFeatures/GridDictionary2.csv "
       "(official published size: 3120 cases; 9360 = total source sims)")
    csv_path = DATA / "NE39-Open" / "PowerGridFeatures" / "GridDictionary2.csv"
    if not csv_path.exists():
        print(f"  MISSING: {csv_path}")
        return False
    df = pd.read_csv(csv_path)
    print(f"  shape: {df.shape}")
    if "Stability" in df.columns:
        print(f"  Stability dist: {dict(Counter(df['Stability'].tolist()))}")
    print(f"  NOTE: Zenodo 7350829 metadata: 3120-case stochastic subset drawn from "
          f"9360 systematic sims (70%/20%/10% 1P/2P/3P faults, <20% unstable). "
          f"Paper Sec 4.1 must be corrected accordingly.")
    return df.shape[0] == 3120


def main() -> int:
    results = {}
    results["IEEE39-Sim"] = check_sim_pkl(
        DATA / "IEEE39-Sim_生成脚本" / "IEEE39_Sim_2400.pkl", 2400)
    results["IEEE118-Sim"] = check_sim_pkl(
        DATA / "IEEE39-Sim_生成脚本" / "IEEE118_Sim_1200.pkl", 1200)
    results["PF-39"] = check_pf39()
    results["NE39-Open"] = check_ne39()

    hr("SUMMARY")
    all_ok = True
    for name, ok in results.items():
        mark = "PASS" if ok else "FAIL"
        all_ok &= ok
        print(f"  {name:<12} {mark}")
    print("\nAll datasets PASS." if all_ok else "\nSome datasets FAILED — see details above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
