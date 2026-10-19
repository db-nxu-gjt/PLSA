# -*- coding: utf-8 -*-
"""统一训练器：相同早停协议 / 多任务损失组装 / 评估与吞吐基准。

所有 torch 模型 forward 返回 dict：
  {"logits": (B,2)}，多任务模型额外含
  {"traj_delta": (B,T,N), "traj_omega": (B,T,N), "phys_loss": 标量}
损失 = CE + λt·masked_MSE(δ,ω) + λp·phys_loss（λt=0.5, λp=0.1，与论文一致）。
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RESULTS = Path(__file__).resolve().parent.parent / "results"
LAM_T, LAM_P, WARMUP_EP = 0.5, 0.1, 10


def batch_loss(model, batch, device, epoch=None, lam_t=LAM_T, lam_p=LAM_P):
    x, desc, y, td, tw, tm = [t.to(device) for t in batch]
    out = model(x, desc)
    ce = F.cross_entropy(out["logits"], y)
    total = ce
    parts = dict(ce=ce.item())
    if "traj_delta" in out:
        m = tm.unsqueeze(-1)
        mse_d = (((out["traj_delta"] - td) ** 2) * m).sum() / (m.sum() * td.shape[-1] + 1e-9)
        mse_w = (((out["traj_omega"] - tw) ** 2) * m).sum() / (m.sum() * tw.shape[-1] + 1e-9)
        traj = mse_d + mse_w
        total = total + lam_t * traj
        parts["traj"] = traj.item()
        if "phys_loss" in out and out["phys_loss"] is not None:
            lam = lam_p * (1.0 if epoch is None else min(epoch / WARMUP_EP, 1.0))
            total = total + lam * out["phys_loss"]
            parts["phys"] = float(out["phys_loss"])
    return total, parts


class Trainer:
    def __init__(self, max_epochs=300, patience=30, lr=1e-3, weight_decay=1e-4,
                 device=None, verbose=False, lam_t=LAM_T, lam_p=LAM_P, cosine=False):
        self.max_epochs, self.patience = max_epochs, patience
        self.lr, self.wd = lr, weight_decay
        self.lam_t, self.lam_p = lam_t, lam_p
        self.cosine = cosine
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

    def fit(self, model, train_loader, val_loader, seed=0):
        torch.manual_seed(seed); np.random.seed(seed)
        model.to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs) \
            if self.cosine else None
        best_val, best_state, best_ep, bad = float("inf"), None, 0, 0
        t0 = time.time()
        for ep in range(1, self.max_epochs + 1):
            model.train()
            for batch in train_loader:
                loss, _ = batch_loss(model, batch, self.device, epoch=ep,
                                     lam_t=self.lam_t, lam_p=self.lam_p)
                if not torch.isfinite(loss):    # 发散场景梯度爆炸保护
                    opt.zero_grad()
                    continue
                opt.zero_grad(); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gn):  # NaN/Inf 梯度守卫
                    opt.zero_grad()
                    continue
                opt.step()
            if sched is not None:
                sched.step()
            model.eval()
            with torch.no_grad():
                vl = np.mean([batch_loss(model, b, self.device, epoch=ep,
                                         lam_t=self.lam_t, lam_p=self.lam_p)[0].item()
                              for b in val_loader])
            if vl < best_val - 1e-5:
                best_val, best_ep, bad = vl, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and ep % 20 == 0:
                print(f"    ep{ep:3d} val={vl:.4f} best={best_val:.4f}")
        if best_state is not None:
            model.load_state_dict(best_state)
        return dict(best_val=float(best_val), best_epoch=best_ep, fit_s=round(time.time() - t0, 1))

    @torch.no_grad()
    def predict(self, model, loader):
        model.eval()
        ys, ps, ts = [], [], []
        for batch in loader:
            x, desc, y, td, tw, tm = [t.to(self.device) for t in batch]
            out = model(x, desc)
            ps.append(F.softmax(out["logits"], dim=1)[:, 1].cpu())
            ys.append(y.cpu())
            ts.append((out.get("traj_delta"), out.get("traj_omega"), td, tw, tm))
        y = torch.cat(ys).numpy()
        p = torch.cat(ps).numpy()
        traj = None
        if ts[0][0] is not None:
            traj = tuple(torch.cat([t[i] for t in ts]) for i in range(5))
        return y, p, traj

    @torch.no_grad()
    def benchmark(self, model, X, n_single=100, batch=512):
        """单样本时延(ms, batch=1) 与吞吐(samples/s, batch=512)。"""
        model.eval()
        dev = next(model.parameters()).device
        x = torch.from_numpy(X[:batch].astype(np.float32)).to(dev)
        desc = torch.zeros(len(x), 5, device=dev)
        for _ in range(5):
            model(x, desc)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_single):
            model(x[:1], desc[:1])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / n_single * 1000
        t0 = time.perf_counter()
        n_done = 0
        for i in range(0, len(x), batch):
            model(x[i:i + batch], desc[i:i + batch]); n_done += len(x[i:i + batch])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        thr = n_done / max(time.perf_counter() - t0, 1e-9)
        return dict(latency_ms=round(lat, 3), throughput_sps=round(thr, 1))


def save_result(res: dict, subdir: str):
    p = RESULTS / subdir
    p.mkdir(parents=True, exist_ok=True)
    name = f"{res['dataset']}_{res['tag']}_seed{res['seed']}.json"
    with open(p / name, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return p / name
