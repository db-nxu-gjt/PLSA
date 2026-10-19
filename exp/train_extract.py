# -*- coding: utf-8 -*-
"""P1-7 训练 + 权重/调制提取：orig 与 cond 变体，协议与论文主结果一致
（d128L3+aug、cosine 3e-4、500/50、lam_t=1.0、lam_p=0.1）。

输出：
  results/pimtnet_cond/   cond 变体 JSON+probs（schema 同 train.py）
  results/pimtnet_params/ weights npz：orig/cond 逐种子
                          {M,D,Pm(softplus后), param_head W/b, cm_test, desc_test}
orig 只存权重不写 JSON（论文主结果目录不动）。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from common.dataset import make_loaders
from common.trainer import Trainer
from common.metrics import cls_metrics
from models.pimtnet import PIMTNet

ROOT = Path(__file__).resolve().parent
SEEDS = range(5)
ARCH = dict(d=128, heads=8, n_enc=3, n_dec=3, ff=256, augment=True)
TR = dict(cosine=True, lr=3e-4, max_epochs=500, patience=50, lam_t=1.0)


def extract(model, desc_te, out_npz):
    M = torch.nn.functional.softplus(model.M).detach().cpu().numpy()
    D = torch.nn.functional.softplus(model.D).detach().cpu().numpy()
    Pm = model.Pm.detach().cpu().numpy()
    d = dict(M=M, D=D, Pm=Pm)
    if getattr(model, "cond_params", False):
        with torch.no_grad():
            cm = 1.0 + 0.5 * torch.tanh(model.param_head(desc_te.to(next(model.parameters()).device)))
        d["cm"] = cm.cpu().numpy()                      # (N,3,n_mach)
        d["param_head_W"] = model.param_head.weight.detach().cpu().numpy()
        d["param_head_b"] = model.param_head.bias.detach().cpu().numpy()
    np.savez(out_npz, **d)


def main():
    loaders, meta = make_loaders("ieee39", "main")
    # 测试集 desc（与 probs 对齐的顺序）
    desc_te = loaders["test"].dataset.tensors[1]
    out_json = ROOT / "results" / "pimtnet_cond"
    out_w = ROOT / "results" / "pimtnet_params"
    out_json.mkdir(exist_ok=True)
    out_w.mkdir(exist_ok=True)

    for variant, cond in [("orig", False), ("cond", True)]:
        for seed in SEEDS:
            wpath = out_w / f"{variant}_seed{seed}.npz"
            jpath = out_json / f"ieee39_main_seed{seed}.json"
            if wpath.exists() and (variant == "orig" or jpath.exists()):
                print(f"[skip] {variant} seed{seed}")
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = PIMTNet(meta["n_ch"], meta["n_desc"], meta["n_mach"],
                            **{**ARCH, "cond_params": cond})
            tr = Trainer(**TR)
            fit = tr.fit(model, loaders["train"], loaders["val"], seed=seed)
            y, p, _ = tr.predict(model, loaders["test"])
            m = cls_metrics(y, p)
            extract(model, desc_te, wpath)
            if cond:
                np.savez(out_json / f"ieee39_main_seed{seed}_probs.npz", y=y, p=p)
                rec = dict(model="pimtnet_cond", dataset="ieee39", tag="main", seed=seed,
                           params=int(sum(q.numel() for q in model.parameters())),
                           fit_s=round(time.time() - t0, 1), **{k: fit[k] for k in
                                                               ("best_val", "best_epoch")},
                           **{k: round(float(v), 6) for k, v in m.items()})
                jpath.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(f"[done] {variant} seed{seed} acc={m['acc']:.4f} f1={m['f1']:.4f} "
                  f"auc={m['auc']:.4f} best_val={fit['best_val']:.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
