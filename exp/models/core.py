# -*- coding: utf-8 -*-
"""纯 torch 基线模型（分类）：MLP / CNN-1D / LSTM / Transformer / DenseNet-1D。

统一接口：forward(x: (B,T,C), desc: (B,5)) -> {"logits": (B,2)}
"""
import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_ch, n_desc, T=13, n_cls=2, hidden=(256, 128)):
        super().__init__()
        d_in = T * n_ch + n_desc
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2)]
            d = h
        layers.append(nn.Linear(d, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x, desc):
        z = torch.cat([x.flatten(1), desc], dim=1)
        return {"logits": self.net(z)}


class CNN1D(nn.Module):
    def __init__(self, n_ch, n_desc, T=13, n_cls=2, d=64):
        super().__init__()
        self.desc_fc = nn.Linear(n_desc, 32)
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch, d, 3, padding=1), nn.BatchNorm1d(d), nn.ReLU(),
            nn.Conv1d(d, d, 3, padding=1), nn.BatchNorm1d(d), nn.ReLU(),
            nn.Conv1d(d, d, 3, padding=1), nn.BatchNorm1d(d), nn.ReLU())
        self.head = nn.Linear(d + 32, n_cls)

    def forward(self, x, desc):
        z = self.conv(x.transpose(1, 2)).mean(dim=2)
        return {"logits": self.head(torch.cat([z, self.desc_fc(desc)], dim=1))}


class LSTM(nn.Module):
    def __init__(self, n_ch, n_desc, T=13, n_cls=2, d=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, d, layers, batch_first=True, dropout=0.2)
        self.desc_fc = nn.Linear(n_desc, 32)
        self.head = nn.Linear(d + 32, n_cls)

    def forward(self, x, desc):
        _, (h, _) = self.lstm(x)
        z = h[-1]
        return {"logits": self.head(torch.cat([z, self.desc_fc(desc)], dim=1))}


class PosEnc(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1]]


class TransformerCls(nn.Module):
    """PIMT-Net 去掉轨迹任务与物理损失的退化版（消融复用）。"""

    def __init__(self, n_ch, n_desc, T=13, n_cls=2, d=64, heads=4, layers=2, ff=128):
        super().__init__()
        self.inp = nn.Linear(n_ch, d)
        self.pos = PosEnc(d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.desc_fc = nn.Linear(n_desc, 32)
        self.head = nn.Linear(d + 32, n_cls)

    def forward(self, x, desc):
        h = self.enc(self.pos(self.inp(x))).mean(dim=1)
        return {"logits": self.head(torch.cat([h, self.desc_fc(desc)], dim=1))}


class DenseBlock(nn.Module):
    def __init__(self, n_in, growth, n_layers, k=3):
        super().__init__()
        self.layers = nn.ModuleList()
        d = n_in
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(d), nn.ReLU(),
                nn.Conv1d(d, growth, k, padding=k // 2)))
            d += growth

    def forward(self, x):
        for l in self.layers:
            x = torch.cat([x, l(x)], dim=1)
        return x


class DenseNet1D(nn.Module):
    """1D DenseNet 适配短序列（LLM-NND 的人工设计基线）。"""

    def __init__(self, n_ch, n_desc, T=13, n_cls=2, d0=64, growth=32):
        super().__init__()
        self.stem = nn.Conv1d(n_ch, d0, 3, padding=1)
        self.b1 = DenseBlock(d0, growth, 4)
        self.t1 = nn.Conv1d(d0 + 4 * growth, d0, 1)
        self.b2 = DenseBlock(d0, growth, 4)
        d_out = d0 + 4 * growth
        self.desc_fc = nn.Linear(n_desc, 32)
        self.head = nn.Linear(d_out + 32, n_cls)

    def forward(self, x, desc):
        z = self.b2(self.t1(self.b1(self.stem(x.transpose(1, 2))))).mean(dim=2)
        return {"logits": self.head(torch.cat([z, self.desc_fc(desc)], dim=1))}


BUILDERS = {"mlp": MLP, "cnn1d": CNN1D, "lstm": LSTM,
            "transformer": TransformerCls, "densenet": DenseNet1D}
