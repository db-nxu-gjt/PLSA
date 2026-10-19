# -*- coding: utf-8 -*-
"""PIMT-Net：本文方法（论文 3.1 节）。

结构 = Transformer 编码器 + 可学习查询解码器：
  - 编码器输入：逐机线性投影 (n_ch->d) + 场景量拼接
  - 解码器：24 个可学习查询 token 交叉注意编码器记忆 -> 逐机 δ/ω 头
多任务损失（由 trainer 组装）：
  L = CE + 0.5·MSE(δ,ω 未来24步) + 0.1·(||r_sw/ω_b||² + ||r_kin/ω_b||²)
物理残差作用于预测轨迹：
  - 运动学: dδ/dt = ω_b·(ω - 1)   （ω 为 pu 转速）
  - 摇摆:  M·d²δ/dt² + D·dδ/dt = ω_b·(Pm - Pe)
  其中 M/D/Pm 为逐机可学习参数（softplus 保证正性，物理约束学习），
  Pe 在预测窗取观测窗末两步线性外插值（与论文 PMU 外插口径一致）。
"""
import torch
import torch.nn as nn

from .core import PosEnc

WB = 2 * 3.141592653589793 * 60.0   # ω_b = 377 rad/s
FS = 60.0


class PIMTNet(nn.Module):
    def __init__(self, n_ch, n_desc, n_mach, d=64, heads=4, n_enc=2, n_dec=2, ff=128,
                 T_pred=24, dropout=0.1, augment=False, use_traj_head=True,
                 cond_params=False):
        super().__init__()
        self.n_mach, self.T_pred = n_mach, T_pred
        self.augment = augment
        self.use_traj_head = use_traj_head   # 消融开关：False = 去轨迹头（纯分类塔）
        self.cond_params = cond_params       # 场景条件化参数：M/D/Pm 由 desc 经小头调制
        self.inp = nn.Linear(n_ch, d)
        self.desc_fc = nn.Linear(n_desc, d)
        self.pos = PosEnc(d)
        enc_layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_enc)
        if use_traj_head:
            self.query = nn.Parameter(torch.randn(T_pred, d) * 0.02)
            dec_layer = nn.TransformerDecoderLayer(d, heads, ff, dropout=dropout, batch_first=True)
            self.dec = nn.TransformerDecoder(dec_layer, n_dec)
            self.head_d = nn.Linear(d, n_mach)
            self.head_w = nn.Linear(d, n_mach)
            cls_in = d + d
        else:
            cls_in = d
        self.head_cls = nn.Linear(cls_in, 2)
        if use_traj_head:
            with torch.no_grad():   # ω 初值≈1 pu（同步转速），避免运动学残差初期爆炸
                self.head_w.bias.fill_(1.0)
        # 物理参数（逐机，可学习，正性）——仅在保留轨迹头时参与（物理残差作用于预测轨迹）
        self.M = nn.Parameter(torch.ones(n_mach) * 0.1)
        self.D = nn.Parameter(torch.ones(n_mach) * 0.1)
        self.Pm = nn.Parameter(torch.ones(n_mach))
        if cond_params and use_traj_head:
            # 场景条件化调制头：desc -> 逐场景 M/D/Pm 调制系数（(0.5,1.5)，零初始化=恒等）
            self.param_head = nn.Linear(n_desc, 3 * n_mach)
            nn.init.zeros_(self.param_head.weight)
            nn.init.zeros_(self.param_head.bias)

    def forward(self, x, desc):
        B, T, _ = x.shape
        if self.training and self.augment:
            # 输入增广（标准化域）：小幅幅值抖动 + 高斯量测噪声，缓解小样本过拟合
            x = x * (1.0 + 0.02 * torch.randn_like(x)) + 0.01 * torch.randn_like(x)
        h = self.enc(self.pos(self.inp(x) + self.desc_fc(desc).unsqueeze(1)))
        if self.use_traj_head:
            q = self.query.unsqueeze(0).expand(B, -1, -1)
            z = self.dec(q, h)                                   # (B,T_pred,d)
            pred_d = self.head_d(z)                              # (B,T_pred,n_mach)
            pred_w = self.head_w(z)
            cls_z = torch.cat([h.mean(dim=1), z.mean(dim=1)], dim=1)
            out = {"logits": self.head_cls(cls_z), "traj_delta": pred_d,
                   "traj_omega": pred_w, "phys_loss": None}
            if self.training:
                M_b = D_b = Pm_b = None
                if self.cond_params:
                    cm = 1.0 + 0.5 * torch.tanh(self.param_head(desc))  # (B,3*n_mach)
                    cm = cm.view(B, 3, self.n_mach)
                    M_b = nn.functional.softplus(self.M).unsqueeze(0) * cm[:, 0]
                    D_b = nn.functional.softplus(self.D).unsqueeze(0) * cm[:, 1]
                    Pm_b = self.Pm.unsqueeze(0) * cm[:, 2]
                out["phys_loss"] = self._physics(x, pred_d, pred_w, M_b, D_b, Pm_b)
        else:
            # 消融：无轨迹头时省略 traj_* 键 → trainer 自动跳过轨迹/物理损失
            out = {"logits": self.head_cls(h.mean(dim=1))}
        return out

    def _physics(self, x, pred_d, pred_w, M_b=None, D_b=None, Pm_b=None):
        dt = 1.0 / FS
        # Pe 线性外插：观测窗末两步斜率延续到预测窗
        pe_last, pe_prev = x[:, -1, -self.n_mach:], x[:, -2, -self.n_mach:]
        slope = (pe_last - pe_prev) / dt
        t_h = (torch.arange(self.T_pred, device=x.device, dtype=x.dtype) + 1) * dt
        pe_ext = pe_last.unsqueeze(1) + slope.unsqueeze(1) * t_h.unsqueeze(0).unsqueeze(-1)
        # 一阶/二阶差分（预测窗内部，丢首步）
        d_dot = (pred_d[:, 1:] - pred_d[:, :-1]) / dt
        d_ddot = (pred_d[:, 2:] - 2 * pred_d[:, 1:-1] + pred_d[:, :-2]) / dt ** 2
        w_mid = pred_w[:, 1:-1]
        pe_mid = pe_ext[:, 1:-1]
        r_kin = (d_dot[:, 1:] - WB * (w_mid - 1.0)) / WB      # 运动学残差（按论文除以 ω_b）
        M = M_b if M_b is not None else nn.functional.softplus(self.M).expand(x.shape[0], -1)
        D = D_b if D_b is not None else nn.functional.softplus(self.D).expand(x.shape[0], -1)
        Pm = Pm_b if Pm_b is not None else self.Pm.expand(x.shape[0], -1)
        r_sw = (M.unsqueeze(1) * d_ddot + D.unsqueeze(1) * d_dot[:, 1:]
                - WB * (Pm.unsqueeze(1) - pe_mid)) / WB       # 摇摆残差（归一化）
        return (r_kin ** 2).mean() + (r_sw ** 2).mean()
