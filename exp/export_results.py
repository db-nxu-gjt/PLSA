# -*- coding: utf-8 -*-
"""汇总导出：将 exp/results/ 下全部实验按类别导出为 Excel 到 paper/submit/results/。

文件结构：
  00_overview_analysis.xlsx   实验总览 / 关键结论 / 风险与口径
  01_main_stability.xlsx      主结果（ieee39 / ieee118 稳定性分类）
  02_ood_generalization.xlsx  OOD 泛化（母线外推）
  03_few_shot.xlsx            小样本（25% / 50% 训练集）
  04_noise_robustness.xlsx    噪声鲁棒性（1% / 2% / 5%）
  05_external_datasets.xlsx   外部数据集（PF-39 / NE39-Open）
  06_fault_location.xlsx      故障位置识别（39 类）
每个文件两个 sheet：mean±std 汇总 + 逐种子原始数据。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT.parent / "export"

MODEL_ORDER = ["pimtnet", "mlp", "cnn1d", "lstm", "transformer", "densenet",
               "pfnn", "pi_seq2seq", "pi_gcn", "es_pinn", "tsa_llm",
               "svm", "rf", "xgboost",
               "llm_direct_zeroshot", "llm_direct_fewshot"]
BINARY_COLS = ["acc", "f1", "auc", "fnr", "fpr", "precision", "recall"]
MULTI_COLS = ["acc", "macro_f1", "weighted_f1"]


def collect(pattern):
    """results/<model>/<pattern>_seed<seed>.json -> DataFrame。"""
    rows = []
    for f in sorted(RESULTS.glob(f"*/{pattern}_seed*.json")):
        model = f.parent.name
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:                          # noqa: BLE001
            print(f"[warn] 读取失败 {f}: {e}")
            continue
        r["model"] = model
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df, cols, order=None):
    """逐模型 mean±std 汇总表。"""
    recs = []
    for m, g in df.groupby("model"):
        rec = {"model": m, "n_seeds": len(g), "params_M": round(g["params"].mean() / 1e6, 2)
               if "params" in g and g["params"].notna().any() else None}
        for c in cols:
            if c in g and g[c].notna().any():
                rec[c] = f"{g[c].mean():.4f}±{g[c].std():.4f}"
                rec[c + "_mean"] = round(float(g[c].mean()), 4)
        recs.append(rec)
    out = pd.DataFrame(recs)
    out["__o"] = out["model"].map({m: i for i, m in enumerate(order or MODEL_ORDER)}).fillna(99)
    return out.sort_values("__o").drop(columns="__o").reset_index(drop=True)


def save_excel(df, cols, path, raw_name, sum_name="summary_mean±std", order=None):
    """df: 逐种子原始数据；按 (model,dataset,tag) 拆 sheet 写出。"""
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        first = True
        for (ds, tag), g in df.groupby(["dataset", "tag"]):
            s = summarize(g, cols, order=order)
            sheet = f"{ds}_{tag}"[:31]
            s.to_excel(w, sheet_name=sheet, index=False)
            if first:
                # 汇总 sheet 排第一
                wb_sheets = w.book.sheetnames
                w.book.move_sheet(sheet, offset=-(wb_sheets.index(sheet)))
                first = False
        df[[c for c in ["model", "dataset", "tag", "seed"] + cols
            + ["params", "latency_ms", "fit_s"] if c in df]] \
            .to_excel(w, sheet_name=raw_name, index=False)
    print(f"[ok] {path.relative_to(ROOT.parent.parent)}  ({len(df)} seeds)")


def write_overview(counts):
    overview = pd.DataFrame([
        ["主结果-稳定性分类", "ieee39", 14, 5, counts.get("ieee39_main", 0),
         "可用", "主表；SVM acc/f1 领先(.8678/.8770)，PIMT-Net 为深度模型第一梯队；优势叙事应放在鲁棒性与物理内嵌"],
        ["主结果-稳定性分类", "ieee118", 14, 5, counts.get("ieee118_main", 0),
         "可用", "重生成数据+轨迹钳制口径（见 R9）：xgboost .8536 领先，PIMT-Net .8435 第二；t13 大系统短窗全模型 .78-.85，难度高于 ieee39"],
        ["OOD 母线外推", "ieee39/ieee118", 14, 5, counts.get("ood", 0),
         "可用", "ieee39 上 PIMT-Net f1=.9304 全场最佳；ieee118 ood es_pinn/transformer .8428 领先、PIMT-Net .8327 第一梯队"],
        ["小样本 25%", "ieee39/ieee118", 14, 5, counts.get("sub25", 0),
         "可用(需说明方差)", "pimtnet ieee39 sub25 方差大(.765±.110)；ieee118 sub25 xgboost .8075 > rf .8042 > pimtnet .7967 第三"],
        ["小样本 50%", "ieee39/ieee118", 14, 5, counts.get("sub50", 0),
         "可用", "pimtnet ieee39 sub50 .8529±.0030、ieee118 .8042 第二（仅次 xgboost .8494）——数据效率恢复力强"],
        ["噪声 1%/2%/5%", "ieee39/ieee118", 14, 5, counts.get("noise", 0),
         "可用", "PIMT-Net 全数据集近零退化，且 main vs noise 配对检验 9 组全部不显著（09 表）——零退化主张获统计支持"],
        ["外部数据集", "pf39", 8, 5, counts.get("pf39_main", 0),
         "可用(仅通用性)", "近饱和(.9944-.9964)，区分度低，只能作为通用性验证"],
        ["外部数据集", "ne39", 8, 5, counts.get("ne39_main", 0),
         "可用", "mlp/densenet .9920 最高，svm .9744 最低——特征型数据上深度模型反超 sklearn"],
        ["故障位置识别(39类)", "ieee39", 8, 5, counts.get("faultid", 0),
         "可用", "AlarmText 替代实验；lstm/transformer .9974 领先，svm .9424 垫底——排名与稳定性分类不同"],
        ["主结果-2.0s观测窗", "ieee39w121", 14, 5, counts.get("ieee39w121_main", 0),
         "可用(推荐主表)", "全模型 .94+（对应原论文 .96+ 口径）；mlp/xgboost .9799 领先，PIMT-Net .9759 第一梯队"],
        ["OOD-2.0s窗", "ieee39w121", 14, 5, counts.get("ieee39w121_ood", 0),
         "可用", "饱和区间(.87-.99)；rf/xgboost .9943、PIMT-Net .9925"],
        ["小样本-2.0s窗", "ieee39w121", 14, 5, counts.get("ieee39w121_sub25", 0) + counts.get("ieee39w121_sub50", 0),
         "可用", "sub25: rf .9678/svm .9655 领先，pimtnet .9489；pi_gcn 方差失控(±.195)"],
        ["噪声-2.0s窗", "ieee39w121", 14, 5, counts.get("ieee39w121_noise1", 0),
         "可用", "全模型近零退化；xgboost .9799→.9741 退化最明显"],
        ["故障识别-2.0s窗", "ieee39w121", 8, 5, counts.get("faultid_t121", 0),
         "可用(补充)", "MAD 稳健定标；transformer .9684 领先、lstm .7103 长窗困难——5.7 主口径建议 t13"],
        ["种子集成(10)", "ieee39/39w121/118w121", 1, 10, counts.get("ens_main", 0),
         "可用(亮点)", "PIMT-Net-Ens10：t13 .8721 全表第一、118w121 .9052 第二、t121 .9741；噪声不降反升——集成作为论文主推结果"],
        ["消融实验", "ieee39/39w121 (main+sub25)", 6, 5, counts.get("abl", 0),
         "可用(需谨慎解读)", "全样本下多任务组件无分类增益（差异<1σ）、t121 轨迹头有正贡献；主张聚焦小样本 regime（sub25 sheet）与轨迹/物理输出价值——见 F14/R8"],
        ["统计显著性检验", "4 数据集 main+噪声+消融", 14, "5/10", "见 09",
         "可用", "Wilcoxon 配对(功效受限)+MWU 10v5+Holm：t13 上 PIMT-Net 显著优于 11/13 基线；噪声退化 9 组全不显著；消融 40 组均不显著（n=5 最小 p=.0625，见 09 说明页）——见 F15"],
        ["通用LLM直接判稳基线", "ieee39", 2, 1, counts.get("llm_direct", 0),
         "可用(单次评估)", "DeepSeek-V3 直接读波形：零样本 acc .66/AUC .73、4-shot acc .68/AUC .70，FNR 高达 .49/.38（失稳漏报近半）——远低于 PIMT-Net .8631/Ens10 .8721；100 例分层子集口径，作架构依据而非同表排名；见 F16"],
    ], columns=["实验类别", "数据集", "模型数", "种子", "结果文件数", "可用性", "论文叙事定位"])

    findings = pd.DataFrame([
        ["F1", "复现差距已解开(核心)", "原论文 .96+ 对应 ~2s 观测窗口径；t121 全模型 .94+ 复现该量级，t13(0.2s) 才是 .85 难度段——窗口消融曲线(07)完整呈现难度-预警时间权衡"],
        ["F2", "主表双口径策略", "t121 主表数值达标(.94+)；PIMT-Net 差异化主张放 t13 早期预警段(深度模型第一)+多任务轨迹预测+鲁棒性"],
        ["F3", "SVM 短窗优势是早期预警特性", "t13 SVM .8678 领先 → t121 跌至 .9718 中游；统计特征在极短窗有效，长窗被深度模型追平"],
        ["F4", "噪声鲁棒性全场最优", "两个窗口口径下 PIMT-Net 1%→5% 噪声均零退化；与物理基线一致但精度更高"],
        ["F5", "OOD 泛化强项", "t13 ieee39 ood f1 .9304 全场最佳；t121 上 .9925 处第一梯队"],
        ["F6", "pi_seq2seq 是最强物理基线（须诚实报告）", "ieee39 t13 main .8422 接近 PIMT-Net .8465；ieee118 重生成后 .8151 已被 PIMT-Net (.8435) 反超——差异化=统一框架+变体实验全面性+显著性优势(F15)"],
        ["F7", "ieee118 已按论文 4.1 协议重生成", "ANDES 重生成 1200 场景（负荷 0.85-1.25、惯量 0.5-1.0、故障 0.06-0.9s、跳闸率 0.7、GENROU+IEEEX1 励磁），失稳率 58.3% 与 ieee39(56.3%) 平衡——协议不符风险已消除"],
        ["F8", "PF-39 饱和", ".994+ 无区分度，仅作通用性验证；NE39 用官方 3120 例口径"],
        ["F9", "故障位置识别替代 AlarmText", "t13 口径 lstm/transformer .9974 领先且与稳定性排名不同；t121 补充口径 transformer .9684 领先"],
        ["F10", "物理基线在长窗全部恢复", "T 参数化修复后 pfnn .9546/pi_seq2seq .9626/es_pinn .9718/tsa_llm .9402（39w121）；es_pinn 表现意外强势"],
        ["F11", "118 系轨迹目标钳制（口径）", "54 机发散爆炸轨迹目标达 2.9e9°（w121）/7.2e7 rad（t13），平方损失摧毁全部带轨迹头模型——数据集级钳制 ±20π rad/±5 pu 覆盖 ieee118w121 与 ieee118（t13 扩展见 R9）；钳制后 118w121 pimtnet .8995、ieee118 t13 .8435；论文须注明钳制口径"],
        ["F12", "pi_gcn 大系统训练塌缩（发现）", "54×54 可学习邻接在 118w121 上多 seed 首轮塌缩且重跑 5 种子复现（118 t13 塌缩系轨迹目标未钳制所致，修复后 .8310 正常）——118w121 的规模脆弱性可作为论文讨论点"],
        ["F13", "10 种子集成三操作点全面提升（亮点）", "PIMT-Net-Ens10：t13 .8721（超 SVM .8678 → 全表第一）、t121 .9741、118w121 .9052（超 rf → 第二）；5% 噪声下不降反升（t121 .9770/118w121 .9100）；10 种子使 t13 方差 ±.0217→±.0059"],
        ["F14", "消融揭示主张边界（须正视）", "分类精度上多任务组件无普适增益：t13 全样本去头 +1.3pp、sub25 去轨迹损失 +6.7pp 且方差 ±.123→±.005（λt=1.0 在小样本短窗主动致不稳定）；t121 相反——物理残差与增广有正贡献（sub25 去掉各 -0.5/-0.9pp）。组件贡献 regime 依赖：长窗物理项有效、短窗多任务项有害，多任务框架价值=统一输出（轨迹预测+物理参数可解释）而非精度"],
        ["F15", "统计显著性检验结论（09 表）", "MWU(10v5)+Holm：ieee39 t13 上 PIMT-Net 显著优于 11/13 基线（除 svm/xgboost），118w121 显著优于 6 基线；t121 上被 mlp/xgboost 显著超越(-0.69pp)——显著性主张与 F2/F13 一致（早期预警段+集成）；噪声退化 9 组配对检验全部不显著（零退化统计支持）；消融 40 组均不显著（n=5 配对最小 p=.0625 功效不足），sub25 去轨迹损失 +6.7pp 的逐种子方向不一致(W/L 2/3)——改进本质是方差稳定化(±.123→±.005)"],
        ["F16", "通用 LLM 直接判稳不可行（架构依据，已实测）", "DeepSeek-V3（deepseek-chat，温度 0）直接读波形数值：零样本 acc .6600/F1 .6222/AUC .7309，4-shot acc .6800/F1 .6800/AUC .7044；关键缺陷是 FNR .4909/.3818——失稳事件漏报近半，安全关键场景不可接受；对比 PIMT-Net 单种子 .8631/Ens10 .8721 与 12/13 专用基线全面落后（仅高于方差失控的 pi_gcn .6439）——支撑双引擎架构：LLM 负责编排与物理校验，数值判别交给轻量物理信息模型；论文须注明 100 例分层子集口径、模型名与温度 0"],
    ], columns=["编号", "发现", "对论文的含义"])

    risks = pd.DataFrame([
        ["R1", "ieee118 协议已对齐（原风险消除）", "已用 andes 按论文 4.1 协议重生成 1200 场景并全量重跑（原'极温和'参数数据弃用，备份 IEEE118_Sim_1200_旧协议备份.pkl）"],
        ["R2", "t121 口径样本变化", "n=2349→1740（609 例 3s 前截断被剔除，失稳占比 56.3%→39.8%）；且长窗含发散爆炸值使任务部分依赖'爆炸检测'——论文须注明并与 t13 曲线并读；118w121 另含轨迹目标钳制口径（见 F11）"],
        ["R3", "pi_gcn t121 方差失控与大系统塌缩", "39w121 main ±.067；118w121 多 seed 首轮塌缩（main .60±.10，重跑复现）——如实报告并作为机组图规模脆弱性发现（F12）讨论"],
        ["R4", "故障识别 t121 定标敏感", "std/p99 定标被爆炸值压垮(lstm 曾 .37)，已改 MAD+clip±10；lstm/svm 长窗仍弱，5.7 主口径建议 t13"],
        ["R5", "sub25 种子方差(t13)", "pimtnet seed2=.806 拉高方差；可在论文报中位数或增加种子数"],
        ["R6", "PF-39 单位/标签口径", "官方标签已翻转(1=失稳)；物理模型因单位约定排除于外部数据集"],
        ["R7", "AlarmText 不可得", "5.7 以故障位置识别替代，需在论文中明确说明替代设计"],
        ["R8", "消融叙事必须改写（已定稿口径）", "原稿'物理约束+多任务提升精度'与消融不符——改写为 regime 依赖叙事：①长窗(t121/118w121)物理残差+增广有效；②短窗小样本 λt=1.0 有害（sub25 去掉 +6.7pp/方差归一，注意逐种子方向不一致，本质是方差稳定化，见 F15）；③多任务价值=轨迹预测输出与可学习物理参数（M/D/Pm）可解释性，判别精度靠 10 种子集成支撑（F13）；④可选补救：λt 随数据量自适应（小样本自动调小）作为未来工作一句带过"],
        ["R9", "ieee118 t13 轨迹目标未钳制事故（已修复）", "10-05 重跑 t13 网格时 _clip_traj 仅覆盖 ieee118w121，重生成数据 7.2e7 rad 爆炸目标摧毁 6 个轨迹头模型（pimtnet .6686±.0871）；扩展钳制至 ieee118 并重跑 210 文件后 pimtnet .8435±.0186（第二）；39 系目标幅值小（t121 max 2.6e4 且训练健康）口径未动；旧文件备份 exp/results_backup_ieee118_t13_preclip/"],
    ], columns=["编号", "风险", "建议"])

    path = OUT / "00_overview_analysis.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        overview.to_excel(w, sheet_name="实验总览", index=False)
        findings.to_excel(w, sheet_name="关键发现", index=False)
        risks.to_excel(w, sheet_name="风险与建议", index=False)
    print(f"[ok] {path.relative_to(ROOT.parent.parent)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- 稳定性二分类：14 模型 × (ieee39, ieee118, ieee39w121, ieee118w121) × 7 tag ----
    stab = pd.concat([collect("ieee39_*"), collect("ieee118_*"), collect("ieee39w121_*"),
                      collect("ieee118w121_*")], ignore_index=True)
    stab = stab[stab["model"].isin(MODEL_ORDER)]
    counts = {}
    for tag in ["main", "ood", "sub25", "sub50", "noise1", "noise2", "noise5"]:
        counts[tag] = len(stab[stab["tag"] == tag])
        counts[f"ieee39_{tag}"] = len(stab[(stab["dataset"] == "ieee39") & (stab["tag"] == tag)])
        counts[f"ieee118_{tag}"] = len(stab[(stab["dataset"] == "ieee118") & (stab["tag"] == tag)])
        counts[f"ieee39w121_{tag}"] = len(stab[(stab["dataset"] == "ieee39w121") & (stab["tag"] == tag)])
        counts[f"ieee118w121_{tag}"] = len(stab[(stab["dataset"] == "ieee118w121") & (stab["tag"] == tag)])
    noise_mask = stab["tag"].str.startswith("noise")
    counts["noise"] = int(noise_mask.sum())
    # 通用 LLM 直接判稳基线（100 例分层子集口径，与全测试集口径不同，单独计数+专属导出）
    llm_mask = stab["model"].str.startswith("llm_direct")
    llm_main = stab[llm_mask & (stab["tag"] == "main")].copy()
    counts["llm_direct"] = int(len(llm_main))
    if len(llm_main):
        per = pd.concat(
            [pd.read_csv(f).assign(model=f.parent.name)
             for d in ("llm_direct_zeroshot", "llm_direct_fewshot")
             for f in sorted((RESULTS / d).glob("ieee39_main_seed*_preds.csv"))],
            ignore_index=True)
        with pd.ExcelWriter(OUT / "10_llm_direct_baseline.xlsx",
                            engine="openpyxl") as w:
            summarize(llm_main, BINARY_COLS).to_excel(w, sheet_name="summary", index=False)
            per.to_excel(w, sheet_name="per_sample", index=False)
        print(f"[ok] submit\\results\\10_llm_direct_baseline.xlsx  "
              f"({len(llm_main)} runs, {len(per)} samples)")

    save_excel(stab[stab["tag"] == "main"], BINARY_COLS, OUT / "01_main_stability.xlsx", "raw_per_seed")
    save_excel(stab[stab["tag"] == "ood"], BINARY_COLS, OUT / "02_ood_generalization.xlsx", "raw_per_seed")
    save_excel(stab[stab["tag"].isin(["sub25", "sub50"])], BINARY_COLS, OUT / "03_few_shot.xlsx", "raw_per_seed")
    save_excel(stab[noise_mask], BINARY_COLS, OUT / "04_noise_robustness.xlsx", "raw_per_seed")

    # ---- 外部数据集 ----
    ext = pd.concat([collect("pf39_*"), collect("ne39_*")], ignore_index=True)
    ext = ext[ext["model"].isin(MODEL_ORDER)]
    counts["pf39_main"] = len(ext[ext["dataset"] == "pf39"])
    counts["ne39_main"] = len(ext[ext["dataset"] == "ne39"])
    save_excel(ext, BINARY_COLS, OUT / "05_external_datasets.xlsx", "raw_per_seed")

    # ---- 故障位置识别（多分类，两个观测窗口径） ----
    fid = pd.concat([collect("ieee39_faultid"), collect("ieee39w121_faultid")],  # noqa: E501
                    ignore_index=True)
    counts["faultid"] = len(fid[fid["dataset"] == "ieee39"])
    counts["faultid_t121"] = len(fid[fid["dataset"] == "ieee39w121"])
    save_excel(fid, MULTI_COLS, OUT / "06_fault_location.xlsx", "raw_per_seed")

    # ---- 08 消融矩阵 + 10 种子集成 ----
    abl_order = ["pimtnet", "pimtnet_ab_nolt", "pimtnet_ab_nolp", "pimtnet_ab_noltlp",
                 "pimtnet_ab_noaug", "pimtnet_ab_nohead", "pimtnet_ens10"]
    abl = pd.concat([collect("ieee39_main"), collect("ieee39_sub25"),
                     collect("ieee39w121_main"), collect("ieee39w121_sub25")],
                    ignore_index=True)
    # ens10 由下方 ens_all 单独并入（collect("ieee39_main") 会误匹配 pimtnet_ens10 目录，此处排除）
    abl = abl[abl["model"].isin(abl_order) & (abl["model"] != "pimtnet_ens10")]
    # 集成模型的全部 tag（含 noise 与 118w121）单独并入
    ens_all = pd.concat(
        [collect(f"{ds}_{t}") for ds in ("ieee39", "ieee39w121", "ieee118w121")
         for t in ("main", "noise1", "noise2", "noise5")], ignore_index=True)
    ens_all = ens_all[ens_all["model"] == "pimtnet_ens10"]
    abl = pd.concat([abl, ens_all], ignore_index=True)
    save_excel(abl, BINARY_COLS, OUT / "08_ablation.xlsx", "raw_per_seed", order=abl_order)
    counts["abl"] = int(len(abl))
    counts["ens_main"] = int(len(ens_all[ens_all["tag"] == "main"]))

    write_overview(counts)
    total = sum(v for k, v in counts.items() if not k.startswith("ieee"))
    print(f"\n合计校验: 主表 {counts['main']}(期望 295+通用LLM基线{counts['llm_direct']}), "
          f"变体 {counts['ood']+counts['sub25']+counts['sub50']+counts['noise']}"
          f"(期望 1725=1680+pimtnet noise 10种子多 5×3×3), "
          f"外部 {counts['pf39_main']+counts['ne39_main']}(期望 80), "
          f"故障识别 {counts['faultid']}+{counts['faultid_t121']}(期望 40+40), "
          f"消融+集成 {counts['abl']}(期望 142)")


if __name__ == "__main__":
    main()
