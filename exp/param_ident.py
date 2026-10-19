# -*- coding: utf-8 -*-
"""P1-7 参数可辨识性验证。

1) 全局参数对照：orig 变体学到的 softplus(M_i) vs ANDES ieee39_full GENROU M
   (=2H) 的 Pearson/Spearman 相关（跨 10 机）。
2) 条件化调制对照：cond 变体的 cm_M/cm_D 调制系数与场景量 m_s 的逐机及
   pooled Pearson 相关（测试集 469 例 × 5 种子）。

输入：
  results/pimtnet_params/orig_seed{0..4}.npz  {M,D,Pm}
  results/pimtnet_params/cond_seed{0..4}.npz  {M,D,Pm,cm,param_head_W,param_head_b}
输出：results/param_ident.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.dataset import make_loaders

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
N_SEEDS = 5


def pearson(a, b):
    return float(stats.pearsonr(a, b)[0])


def main():
    out = {}

    # ---- 基准 M 真值（andes GENROU M = 2H）----
    try:
        import andes
        andes.config_logger(stream_level=50)
        ss = andes.load(andes.get_case("ieee39/ieee39_full.xlsx"), setup=False,
                        no_output=True)
        ss.setup()
        m_base = np.asarray(ss.GENROU.M.v, dtype=float)
        out["M_base_2H"] = np.round(m_base, 4).tolist()
        print(f"[andes] GENROU M (2H) = {np.round(m_base, 3).tolist()}")
    except Exception as e:
        m_base = None
        print(f"[warn] andes unavailable: {e}")

    # ---- orig 全局 M vs 基准（跨机相关，5 种子）----
    orig_rs_p, orig_rs_s = [], []
    for i in range(N_SEEDS):
        f = RESULTS / "pimtnet_params" / f"orig_seed{i}.npz"
        if not f.exists():
            continue
        M = np.load(f)["M"].ravel()          # softplus 后，(n_mach,)
        if m_base is not None and len(M) == len(m_base):
            orig_rs_p.append(pearson(M, m_base))
            orig_rs_s.append(float(stats.spearmanr(M, m_base)[0]))
            print(f"[orig seed{i}] M vs 2H: pearson={orig_rs_p[-1]:.3f} "
                  f"spearman={orig_rs_s[-1]:.3f}")
            print(f"           M = {np.round(M, 3).tolist()}")
    if orig_rs_p:
        out["orig_M_vs_2H"] = dict(pearson=[round(v, 4) for v in orig_rs_p],
                                   spearman=[round(v, 4) for v in orig_rs_s],
                                   pearson_mean=round(float(np.mean(orig_rs_p)), 4),
                                   spearman_mean=round(float(np.mean(orig_rs_s)), 4))

    # ---- cond 调制 vs m_s（测试集，5 种子）----
    _, meta = make_loaders("ieee39", "main")
    desc_te = meta["test_desc"] if "test_desc" in meta else None
    if desc_te is None:
        # 回退：直接取 test loader 第二个张量
        loaders, _ = make_loaders("ieee39", "main")
        desc_te = loaders["test"].dataset.tensors[1].numpy()
    m_s = np.asarray(desc_te, dtype=float)[:, 1]
    n_mach = None
    cm_M_all, cm_D_all = [], []
    for i in range(N_SEEDS):
        f = RESULTS / "pimtnet_params" / f"cond_seed{i}.npz"
        if not f.exists():
            continue
        cm = np.load(f)["cm"]                # (N, 3*n_mach) 展平
        cm = cm.reshape(cm.shape[0], 3, -1)  # 与模型 view(B,3,n_mach) 一致
        n_mach = cm.shape[2]
        cm_M_all.append(cm[:, 0, :])         # M 调制
        cm_D_all.append(cm[:, 1, :])         # D 调制
    if cm_M_all:
        cmM = np.concatenate(cm_M_all, axis=0)   # (5N, n_mach)
        cmD = np.concatenate(cm_D_all, axis=0)
        ms_rep = np.tile(m_s, len(cm_M_all))     # (5N,)
        r_M = [pearson(cmM[:, j], ms_rep) for j in range(n_mach)]
        r_D = [pearson(cmD[:, j], ms_rep) for j in range(n_mach)]
        out["cond_modulation_vs_ms"] = dict(
            n_per_seed=int(cmM.shape[0] // len(cm_M_all)), n_seeds=len(cm_M_all),
            r_M_per_machine=[round(v, 4) for v in r_M],
            r_D_per_machine=[round(v, 4) for v in r_D],
            r_M_pooled=round(pearson(cmM.ravel(), np.repeat(ms_rep, n_mach)), 4),
            r_D_pooled=round(pearson(cmD.ravel(), np.repeat(ms_rep, n_mach)), 4),
            cm_M_mean=round(float(cmM.mean()), 4), cm_M_std=round(float(cmM.std()), 4),
            cm_D_mean=round(float(cmD.mean()), 4), cm_D_std=round(float(cmD.std()), 4))
        print(f"[cond] r(cm_M, m_s) per machine = {np.round(r_M, 3).tolist()}")
        print(f"[cond] pooled r_M={out['cond_modulation_vs_ms']['r_M_pooled']}, "
              f"r_D={out['cond_modulation_vs_ms']['r_D_pooled']}")

    (RESULTS / "param_ident.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[ok] results/param_ident.json")


if __name__ == "__main__":
    main()
