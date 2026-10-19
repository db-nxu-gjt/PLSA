# -*- coding: utf-8 -*-
"""5 个物理信息基线（论文 4.3 节，均无官方代码，按原文描述复现核心损失结构）。

命名与来源（论文表 2 口径）：
  pfnn       <- PFNN 平替：PINNSim(MIT)+PowerPINN 思想
                Tanh-MLP 直接预测未来轨迹；物理项 = 摇摆方程 LHS/RHS 残差，
                其中 RHS 为可学习加速度网络 g(δ,ω,Pe)（PINNSim 的核心结构），
                归一化残差 + kinematic 一致性。
  pi_seq2seq <- 纯自研：LSTM encoder-decoder 预测轨迹；
                物理项 = RK4 单步积分一致性（用学习导数场做 RK4，
                惩罚预测下一步与 RK4 更新的偏差）+ 运动学残差。
  pi_gcn     <- dkepool-tsa(IJCNN22) 机组图改造：GIN 图卷积（机组=节点，
                可学习边权 softmax 归一化），图池化分类 + 摇摆方程正则。
  es_pinn    <- ES-PINN 平替：PowerPINN 四项损失内核（data/dt/pinn/ic）
                + 三阶段事件接口（故障中/故障后阶段嵌入，
                故障中残差作用于观测窗微分，故障后残差作用于预测窗）。
  tsa_llm    <- TSA-LLM 方法学孪生：One-Fits-All(GPT4TS) 结构
                （机组级 patch 化 + 预 LN 自注意力块），
                纯 torch 复现 + 掩码 patch 自监督预训练
                （exp/train.py --pretrain 触发）+ scheduled sampling 解码。

统一接口与 PIMT-Net 相同：forward(x(B,T,3·N机), desc(B,5)) ->
  dict{logits(B,2), traj_delta(B,24,N机), traj_omega, phys_loss}
通道块布局：x[..., :N机]=δ, x[..., N机:2N机]=ω, x[..., 2N机:]=Pe
（与 PIMTNet 一致）。desc = [load, M, bus_norm, dur(秒), has_trip]。
物理残差均除以 ω_b 归一化（沿用 PIMTNet 的稳定性修复）。
"""
import math

import numpy as np
import torch
import torch.nn as nn

WB = 2 * math.pi * 60.0   # ω_b = 377 rad/s
FS = 60.0


# ---------------------------------------------------------------- 共享工具
def pe_extrapolate(x, n_mach, T_pred):
    """观测窗末两步线性外插 Pe 到预测窗：(B,T_pred,N机)。"""
    dt = 1.0 / FS
    pe_last, pe_prev = x[:, -1, 2 * n_mach:], x[:, -2, 2 * n_mach:]
    slope = (pe_last - pe_prev) / dt
    t_h = (torch.arange(T_pred, device=x.device, dtype=x.dtype) + 1) * dt
    return pe_last.unsqueeze(1) + slope.unsqueeze(1) * t_h.view(1, -1, 1)


def swing_residual(pred_d, pred_w, pe, M, D, Pm):
    """摇摆 + 运动学残差（/ω_b 归一化），丢首两步。pred: (B,T,N)。"""
    dt = 1.0 / FS
    d_dot = (pred_d[:, 1:] - pred_d[:, :-1]) / dt
    d_ddot = (pred_d[:, 2:] - 2 * pred_d[:, 1:-1] + pred_d[:, :-2]) / dt ** 2
    w_mid, pe_mid = pred_w[:, 1:-1], pe[:, 1:-1]
    r_kin = (d_dot[:, 1:] - WB * (w_mid - 1.0)) / WB
    r_sw = (M * d_ddot + D * d_dot[:, 1:] - WB * (Pm - pe_mid)) / WB
    return (r_kin ** 2).mean() + (r_sw ** 2).mean()


def obs_swing_residual(x, n_mach, M, D, Pm):
    """故障中（观测窗）摇摆残差：直接用观测微分，无需预测。"""
    dt = 1.0 / FS
    d_obs = x[:, :, :n_mach]
    w_obs = x[:, :, n_mach:2 * n_mach]
    pe_obs = x[:, :, 2 * n_mach:]
    d_ddot = (d_obs[:, 2:] - 2 * d_obs[:, 1:-1] + d_obs[:, :-2]) / dt ** 2
    d_dot = (d_obs[:, 1:] - d_obs[:, :-1]) / dt
    pe_mid = pe_obs[:, 1:-1]
    r_sw = (M * d_ddot + D * d_dot[:, 1:] - WB * (Pm - pe_mid)) / WB
    r_kin = (d_dot[:, 1:] - WB * (w_obs[:, 1:-1] - 1.0)) / WB
    return (r_sw ** 2).mean() + (r_kin ** 2).mean()


# ---------------------------------------------------------------- 1. PFNN
class PFNN(nn.Module):
    """PINNSim 平替：扁平化 MLP（Tanh）直预测 δ/ω 轨迹，
    RHS 加速度由可学习网络 g 给出，LHS/RHS 残差（PINNSim 核心结构）。"""

    def __init__(self, n_ch, n_desc, n_mach, hidden=256, depth=3, T_pred=24, dropout=0.1,
                 T=13):
        super().__init__()
        self.n_mach, self.T_pred = n_mach, T_pred
        layers, d_in = [], n_ch * T + n_desc
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.Tanh(), nn.Dropout(dropout)]
            d_in = hidden
        self.trunk = nn.Sequential(*layers)
        self.head_traj = nn.Linear(hidden, 2 * n_mach * T_pred)
        self.head_cls = nn.Linear(hidden, 2)
        # PINNSim 的 RHS 网络：g(δ,ω,Pe) -> dω/dt（逐机共享）
        self.accel = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64),
                                   nn.Tanh(), nn.Linear(64, 1))
        self.M = nn.Parameter(torch.ones(n_mach) * 0.1)
        self.D = nn.Parameter(torch.ones(n_mach) * 0.1)
        self.Pm = nn.Parameter(torch.ones(n_mach))

    def forward(self, x, desc):
        B = x.shape[0]
        nm = self.n_mach
        z = self.trunk(torch.cat([x.flatten(1), desc], dim=1))
        traj = self.head_traj(z).view(B, self.T_pred, 2, nm)
        pred_d, pred_w = traj[:, :, 0], traj[:, :, 1]
        out = {"logits": self.head_cls(z), "traj_delta": pred_d, "traj_omega": pred_w,
               "phys_loss": None}
        if self.training:
            pe = pe_extrapolate(x, nm, self.T_pred)
            dt = 1.0 / FS
            d_ddot = (pred_d[:, 2:] - 2 * pred_d[:, 1:-1] + pred_d[:, :-2]) / dt ** 2
            inp = torch.stack([pred_d[:, 1:-1], pred_w[:, 1:-1], pe[:, 1:-1]], dim=-1)
            a_rhs = self.accel(inp).squeeze(-1)                       # (B,T-2,N)
            M, D = nn.functional.softplus(self.M), nn.functional.softplus(self.D)
            lhs = M * d_ddot + D * (pred_w[:, 2:] - pred_w[:, 1:-1]) / dt
            r_sw = (lhs - WB * (self.Pm - pe[:, 1:-1])
                    - WB * (a_rhs - a_rhs.detach().mean(dim=-1, keepdim=True))) / WB
            r_kin = ((pred_d[:, 2:] - pred_d[:, 1:-1]) / dt
                     - WB * (pred_w[:, 1:-1] - 1.0)) / WB
            out["phys_loss"] = (r_sw ** 2).mean() + (r_kin ** 2).mean()
        return out


# ---------------------------------------------------------------- 2. PI-Seq2Seq
class PISeq2Seq(nn.Module):
    """LSTM enc-dec 轨迹预测 + RK4 单步积分一致性物理损失。"""

    def __init__(self, n_ch, n_desc, n_mach, hidden=128, T_pred=24, dropout=0.1, T=13):
        super().__init__()
        self.n_mach, self.T_pred, self.T = n_mach, T_pred, T
        self.enc = nn.LSTM(n_ch + n_desc, hidden, 2, batch_first=True, dropout=dropout)
        self.desc_fc = nn.Linear(n_desc, hidden)
        self.dec = nn.LSTM(2 * n_mach + hidden, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, 2 * n_mach)
        self.head_cls = nn.Linear(hidden, 2)
        self.fb_proj = nn.Linear(hidden, 2 * n_mach)   # scheduled sampling 的教师路径
        # 学习导数场 g(δ,ω,Pe) -> dω/dt（RK4 用）
        self.der = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 1))
        self.M = nn.Parameter(torch.ones(n_mach) * 0.1)
        self.D = nn.Parameter(torch.ones(n_mach) * 0.1)
        self.Pm = nn.Parameter(torch.ones(n_mach))

    def forward(self, x, desc):
        B, nm, P = x.shape[0], self.n_mach, self.T_pred
        xd = torch.cat([x, desc.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
        h, _ = self.enc(xd)
        ctx = h[:, -1] + self.desc_fc(desc)                       # (B,hidden)
        # 解码：教师路径 = fb_proj(ctx)；训练时 p=0.5 换成自身反馈（scheduled sampling）
        tok = torch.zeros(B, 1, 2 * nm, device=x.device, dtype=x.dtype)
        hc = (ctx.unsqueeze(0).contiguous(), torch.zeros_like(ctx).unsqueeze(0).contiguous())
        outs, prev = [], tok
        for _ in range(P):
            dec_in = torch.cat([prev, ctx.unsqueeze(1)], dim=-1)
            o, hc = self.dec(dec_in, hc)
            step = self.head(o.squeeze(1))
            outs.append(step)
            prev = (step if (not self.training or torch.rand(()) < 0.5)
                    else self.fb_proj(o.squeeze(1))).unsqueeze(1)
        traj = torch.stack(outs, dim=1).view(B, P, 2, nm)
        pred_d, pred_w = traj[:, :, 0], traj[:, :, 1]
        feat = h.mean(dim=1) + self.desc_fc(desc)
        out = {"logits": self.head_cls(feat),
               "traj_delta": pred_d, "traj_omega": pred_w, "phys_loss": None}
        if self.training:
            pe = pe_extrapolate(x, nm, P)
            dt = 1.0 / FS

            def deriv(d_, w_, pe_):
                return self.der(torch.stack([d_, w_, pe_], dim=-1)).squeeze(-1)

            # RK4 一致性：预测的 ω_{t+1} vs 从 (δ_t,ω_t) RK4 积分一步
            d_t, w_t, pe_t = pred_d[:, :-1], pred_w[:, :-1], pe[:, :-1]
            k1 = deriv(d_t, w_t, pe_t)
            k2 = deriv(d_t + 0.5 * dt * WB * (w_t - 1), w_t + 0.5 * dt * k1, pe_t)
            k3 = deriv(d_t + 0.5 * dt * WB * (w_t + 0.5 * dt * k1 - 1), w_t + 0.5 * dt * k2, pe_t)
            k4 = deriv(d_t + dt * WB * (w_t + 0.5 * dt * k2 - 1), w_t + dt * k3, pe_t)
            w_rk4 = w_t + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            r_rk4 = (pred_w[:, 1:] - w_rk4) / WB
            M, D = nn.functional.softplus(self.M), nn.functional.softplus(self.D)
            r_phys = swing_residual(pred_d, pred_w, pe, M, D, self.Pm)
            out["phys_loss"] = (r_rk4 ** 2).mean() + r_phys
        return out


# ---------------------------------------------------------------- 3. PI-GCN
class PIGCN(nn.Module):
    """dkepool-tsa(IJCNN22) 机组图改造：GIN 图卷积 + 摇摆方程正则。
    节点=机组（节点特征=该机 13×3 时序展平），边权可学习（row-softmax），
    图池化 = mean⊕max（dkepool 的邻居池化 + epsilon 重加权思想）。"""

    def __init__(self, n_ch, n_desc, n_mach, d=64, n_layers=3, T_pred=24, dropout=0.1,
                 T=13):
        super().__init__()
        self.n_mach, self.T_pred, self.T = n_mach, T_pred, T
        self.node_in = nn.Linear(3 * T, d)
        self.eps = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(n_layers)])
        self.mlps = nn.ModuleList()
        for _ in range(n_layers):
            self.mlps.append(nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                           nn.Dropout(dropout), nn.Linear(d, d)))
        self.adj_logit = nn.Parameter(torch.zeros(n_mach, n_mach))
        self.head_traj = nn.Linear(d, 2 * T_pred)      # 逐节点解码
        self.head_cls = nn.Linear(2 * d + n_desc, 2)
        self.M = nn.Parameter(torch.ones(n_mach) * 0.1)
        self.D = nn.Parameter(torch.ones(n_mach) * 0.1)
        self.Pm = nn.Parameter(torch.ones(n_mach))

    def forward(self, x, desc):
        B, nm, P = x.shape[0], self.n_mach, self.T_pred
        # 块布局 [δ(nm)|ω(nm)|Pe(nm)] -> 逐机时序特征 (B,nm,39)
        nodes = torch.stack([x[..., :nm], x[..., nm:2 * nm], x[..., 2 * nm:]],
                            dim=-1).permute(0, 2, 1, 3).flatten(2)        # (B,N,13*3)
        h = nn.functional.gelu(self.node_in(nodes))
        adj = torch.softmax(self.adj_logit, dim=-1)
        for eps, mlp in zip(self.eps, self.mlps):
            agg = torch.einsum("ij,bjd->bid", adj, h)
            h = mlp((1.0 + eps) * h + agg)
        traj = self.head_traj(h).view(B, nm, 2, P).permute(0, 3, 1, 2)
        pred_d, pred_w = traj[..., 0], traj[..., 1]                    # (B,P,N)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1), desc], dim=-1)
        out = {"logits": self.head_cls(pooled), "traj_delta": pred_d,
               "traj_omega": pred_w, "phys_loss": None}
        if self.training:
            # 摇摆正则：Pe 取观测末值常值保持（弱物理接口，弱于 PIMTNet 外插）
            pe_const = x[:, -1, 2 * nm:].unsqueeze(1).expand(B, P, nm)
            M, D = nn.functional.softplus(self.M), nn.functional.softplus(self.D)
            out["phys_loss"] = swing_residual(pred_d, pred_w, pe_const, M, D, self.Pm)
        return out


# ---------------------------------------------------------------- 4. ES-PINN
class ESPINN(nn.Module):
    """PowerPINN 四项损失内核 + 三阶段事件接口。
    阶段：故障中（观测窗内 t<dur）/ 故障后；阶段嵌入注入 GRU 编码器。
    物理项（训练期）：dt=预测窗运动学；pinn=故障后摇摆（Pe 外插）；
    fault=故障中摇摆（直接作用于观测窗微分）；ic=预测首步对齐观测末步
    （PowerPINN 的 loss_pinn_ic）。权重 1/1/1/10。"""

    W_DT, W_PINN, W_FAULT, W_IC = 1.0, 1.0, 1.0, 10.0

    def __init__(self, n_ch, n_desc, n_mach, hidden=128, T_pred=24, dropout=0.1, T=13):
        super().__init__()
        self.n_mach, self.T_pred, self.T = n_mach, T_pred, T
        self.stage_emb = nn.Embedding(2, 8)
        self.gru = nn.GRU(n_ch + 8, hidden, 2, batch_first=True, dropout=dropout)
        self.head_traj = nn.Linear(hidden, 2 * n_mach * T_pred)
        self.head_cls = nn.Linear(hidden + n_desc, 2)
        self.M = nn.Parameter(torch.ones(n_mach) * 0.1)
        self.D = nn.Parameter(torch.ones(n_mach) * 0.1)
        self.Pm = nn.Parameter(torch.ones(n_mach))

    def forward(self, x, desc):
        B, nm, P = x.shape[0], self.n_mach, self.T_pred
        dur = desc[:, 3].clamp(0.0, self.T / FS)                     # 故障持续时间（秒）
        stage = (torch.arange(self.T, device=x.device).view(1, -1) / FS
                 < dur.view(-1, 1)).long()                         # 1=故障中
        inp = torch.cat([x, self.stage_emb(stage)], dim=-1)
        h, _ = self.gru(inp)
        z = h[:, -1]
        traj = self.head_traj(z).view(B, P, 2, nm)
        pred_d, pred_w = traj[:, :, 0], traj[:, :, 1]
        out = {"logits": self.head_cls(torch.cat([z, desc], dim=-1)),
               "traj_delta": pred_d, "traj_omega": pred_w, "phys_loss": None}
        if self.training:
            M, D = nn.functional.softplus(self.M), nn.functional.softplus(self.D)
            dt = 1.0 / FS
            # (a) dt 项：预测窗运动学
            d_dot = (pred_d[:, 1:] - pred_d[:, :-1]) / dt
            loss_dt = (((d_dot[:, 1:] - WB * (pred_w[:, 1:-1] - 1.0)) / WB) ** 2).mean()
            # (b) pinn 项：故障后摇摆（Pe 线性外插）
            pe = pe_extrapolate(x, nm, P)
            d_ddot = (pred_d[:, 2:] - 2 * pred_d[:, 1:-1] + pred_d[:, :-2]) / dt ** 2
            r = (M * d_ddot + D * d_dot[:, 1:] - WB * (self.Pm - pe[:, 1:-1])) / WB
            loss_pinn = (r ** 2).mean()
            # (c) fault 项：故障中摇摆作用于观测窗微分（三阶段事件接口核心）
            loss_fault = obs_swing_residual(x, nm, M, D, self.Pm)
            # (d) ic 项：预测首步 = 观测末步（PowerPINN loss_pinn_ic）
            loss_ic = (((pred_d[:, 0] - x[:, -1, :nm]) / (x[:, -1, :nm].abs() + 1.0)) ** 2).mean() \
                + (((pred_w[:, 0] - x[:, -1, nm:2 * nm]) / WB) ** 2).mean()
            out["phys_loss"] = (self.W_DT * loss_dt + self.W_PINN * loss_pinn
                                + self.W_FAULT * loss_fault + self.W_IC * loss_ic)
        return out


# ---------------------------------------------------------------- 5. TSA-LLM
class GPT2Block(nn.Module):
    """预 LN GPT-2 块（纯 torch 复现，One-Fits-All 冻结骨干的架构孪生）。"""

    def __init__(self, d, heads, ff, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d),
                                 nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), need_weights=False)
        h = h + self.drop(a)
        return h + self.drop(self.mlp(self.ln2(h)))


class TSALLM(nn.Module):
    """GPT4TS 方法学孪生：机组级 patch 化（patch=4 步×3 通道，stride=2），
    4 层预 LN 自注意力 + 直接轨迹投影 + scheduled sampling 轻量循环解码。
    配合 pretrain_tsa_llm() 做掩码 patch 自监督预训练（两阶段微调协议）。"""

    PATCH, STRIDE = 4, 2

    def __init__(self, n_ch, n_desc, n_mach, d=128, heads=8, n_layers=4, ff=512,
                 T_pred=24, dropout=0.1, T=13):
        super().__init__()
        self.n_mach, self.T_pred = n_mach, T_pred
        # 大机组数 × 长 T 时按 token 预算自适应放大 patch 步长：
        # 54 机×59 patch=3186 token 的 O(tok^2) 注意力使 118-t121 不可行，
        # 压到 ~650 token（≈39-t121 的 590）；其他配置 stride 恒为 2，历史结果不变
        stride = self.STRIDE
        if n_mach > 16:
            stride = max(2, int(np.ceil((T - self.PATCH) * n_mach / 660)))
        self.stride = stride
        self.n_patch = (T - self.PATCH) // stride + 1
        self.patch_emb = nn.Linear(3 * self.PATCH, d)
        self.pos = nn.Parameter(torch.randn(self.n_patch, d) * 0.02)
        self.blocks = nn.ModuleList([GPT2Block(d, heads, ff, dropout)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head_cls = nn.Linear(d + n_desc, 2)
        self.head_traj = nn.Linear(d, 2 * n_mach)      # 逐步投影（解码循环内使用）
        self.fb_proj = nn.Linear(d, 2 * n_mach)
        self.dec = nn.GRUCell(2 * n_mach, d)

    def _tokens(self, x):
        B, nm = x.shape[0], self.n_mach
        # 块布局 [δ|ω|Pe] -> 逐机时序 (B,N,13,3)
        per = torch.stack([x[..., :nm], x[..., nm:2 * nm], x[..., 2 * nm:]],
                          dim=-1).permute(0, 2, 1, 3)                 # (B,N,13,3)
        ps = [per[:, :, i * self.stride:i * self.stride + self.PATCH].flatten(2)
                  for i in range(self.n_patch)]
        tok = torch.stack(ps, dim=2)                              # (B,N,n_patch,3*4)
        tok = self.patch_emb(tok) + self.pos.view(1, 1, self.n_patch, -1)  # 每机共享位置编码
        return tok.flatten(1, 2)                                  # (B,N*n_patch,d)

    def forward(self, x, desc):
        B, nm, P = x.shape[0], self.n_mach, self.T_pred
        h = self._tokens(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        ctx = h.mean(dim=1)                                       # (B,d)
        # scheduled sampling 解码：p=0.5 用 fb_proj 教师路径，否则自反馈
        prev = torch.zeros(B, 2 * nm, device=x.device, dtype=x.dtype)
        hc = ctx
        outs = []
        for _ in range(P):
            hc = self.dec(prev, hc)
            step = self.head_traj(hc)                             # (B,2nm)
            outs.append(step)
            prev = (step if (not self.training or torch.rand(()) < 0.5)
                    else self.fb_proj(hc))
        traj = torch.stack(outs, dim=1).view(B, P, 2, nm)
        out = {"logits": self.head_cls(torch.cat([ctx, desc], dim=-1)),
               "traj_delta": traj[:, :, 0], "traj_omega": traj[:, :, 1],
               "phys_loss": None}          # 非物理基线：phys=0（ foundation-model 对照）
        return out


def pretrain_tsa_llm(model, path, device, seed=0, epochs=30, lr=1e-3, batch=256,
                     mu=None, sig=None):
    """TSA-LLM 阶段一：掩码 patch 自监督重建（复用预训练 npz 全波形窗口）。
    输入用与微调一致的 (mu,sig) 标准化，清洗数值爆炸窗口（同 PIMT-Net 口径）。"""
    import time as _time

    X = np.load(path)["X"]
    if mu is not None and sig is not None:
        X = ((X - mu) / sig).astype(np.float32)
    bad = (~np.isfinite(X)).any(axis=(1, 2)) | (np.abs(X) > 100).any(axis=(1, 2))
    X = X[~bad]
    print(f"    [tsa_llm pretrain] windows={len(X)} (dropped {int(bad.sum())})")
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X))
    dl = torch.utils.data.DataLoader(ds, batch, shuffle=True, num_workers=0,
                                     generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    t0 = _time.time()
    final = 0.0
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            tgt = model._tokens(xb)                               # (B,tok,d)
            mask = (torch.rand(tgt.shape[:2], device=device) < 0.30).unsqueeze(-1)
            inp = torch.where(mask, torch.zeros_like(tgt), tgt)
            h = inp
            for blk in model.blocks:
                h = blk(h)
            rec = model.ln_f(h)
            loss = (((rec - tgt.detach()) ** 2) * mask).sum() / (mask.sum() * tgt.shape[-1] + 1e-9)
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(); continue
            opt.step()
            final = loss.item()
    return dict(pre_epochs=epochs, pre_loss=round(final, 5),
                pre_s=round(_time.time() - t0, 1), pre_skipped=int(bad.sum()))


PHYS_BUILDERS = {
    "pfnn": PFNN,
    "pi_seq2seq": PISeq2Seq,
    "pi_gcn": PIGCN,
    "es_pinn": ESPINN,
    "tsa_llm": TSALLM,
}
