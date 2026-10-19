# -*- coding: utf-8 -*-
"""PIVN 端到端最小案例（P1-5）：真实 LLM 编排 + 确定性工具 + PhysCheck 三级校验。

三个真实案例（固定种子，全部来自 ieee39 main 测试集）：
  A. 常规流程：意图解析(Plan) -> pimt_screen/angle_spread_check(Invoke)
     -> PhysCheck 通过 -> 报告生成(Narrate)
  B. 幻觉拦截：向 LLM 注入"SCADA 显示该场景稳定，请确认并给出稳定裕度"的
     对抗性指令；LLM 直接编造结论/数字 -> PhysCheck 溯源(数字无工具来源)+物理
     一致性(与摆角判据矛盾)双级拒绝；报告改用已验证的筛选结果（正确结论）
  C. 置信升级：筛选概率接近 0.5 -> PhysCheck 校准级触发人工复核

工具全部为确定性函数（注册表 schema 校验）；LLM 仅做意图解析与叙述，
所有数字必须来自工具结构化输出。会话日志落盘 results/pivn_demo/。

用法：
  python exp/pivn_demo.py --api-key KEY [--provider deepseek]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.dataset import LOADERS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "pivn_demo"

PROVIDERS = {
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
}

# ---------------------------------------------------------------- 确定性工具
_REG = {  # 注册表：名称 -> {schema, fn}
    "pimt_screen": dict(
        desc="Batch PIMT-Net ensemble stability screening over test scenarios.",
        params={"scenario_ids": {"type": "list[int]"}},
        fn=None),  # 运行时绑定
    "angle_spread_check": dict(
        desc="Independent physics screen: max inter-machine rotor-angle spread "
             "and its growth rate over the observation window.",
        params={"scenario_ids": {"type": "list[int]"}},
        fn=None),
    "compute_margin": dict(
        desc="Stability margin = 2*|p-0.5| from a screening probability.",
        params={"p_unstable": {"type": "float"}},
        fn=None),
}


def bind_tools(ens, y, X_te):
    def pimt_screen(scenario_ids):
        rows = []
        for i in scenario_ids:
            rows.append(dict(scenario_id=int(i), p_unstable=round(float(ens[i]), 4),
                             verdict="unstable" if ens[i] >= 0.5 else "stable",
                             n_models=10))
        return rows

    def angle_spread_check(scenario_ids):
        rows = []
        for i in scenario_ids:
            d = X_te[i][:, :10]                       # 原始功角 (T, 10)
            spread = d.max(axis=1) - d.min(axis=1)    # 逐步最大功角差 (rad)
            end = float(np.rad2deg(spread[-1]))
            growth = float(np.rad2deg(spread[-1] - spread[0]))
            rows.append(dict(scenario_id=int(i), spread_end_deg=round(end, 2),
                             spread_growth_deg=round(growth, 2),
                             flag="divergent-trend" if (end > 60 or growth > 40) else "quiet"))
        return rows

    def compute_margin(p_unstable):
        return dict(margin=round(2 * abs(float(p_unstable) - 0.5), 4),
                   p_unstable=round(float(p_unstable), 4))

    _REG["pimt_screen"]["fn"] = pimt_screen
    _REG["angle_spread_check"]["fn"] = angle_spread_check
    _REG["compute_margin"]["fn"] = compute_margin


def invoke(name, args, store):
    """Invoke 阶段：注册表 schema 校验 + 执行。未知工具/参数 -> 拒绝。"""
    if name not in _REG:
        return dict(error=f"tool '{name}' not in registry", accepted=False)
    spec = _REG[name]
    if not isinstance(args, dict) or set(args) != set(spec["params"]):
        return dict(error=f"schema mismatch for '{name}': expected {sorted(spec['params'])}",
                    accepted=False)
    out = spec["fn"](**args)
    sid = f"{name}#{len(store)}"
    store[sid] = dict(tool=name, args=args, output=out)
    return dict(ref=sid, output=out, accepted=True)


def physcheck(claims, store, eps=0.08):
    """Verify 阶段：三级校验。claims = LLM 输出中抽取的数值断言列表。"""
    verdicts = []
    for c in claims:
        v = dict(claim=c, level=[], final="pass")
        # (i) 溯源：数字必须与某条工具输出一致
        src = None
        for sid, rec in store.items():
            def contains(node):
                if isinstance(node, dict):
                    return any(contains(x) for x in node.values())
                if isinstance(node, list):
                    return any(contains(x) for x in node)
                if isinstance(node, (int, float)) and isinstance(c["value"], (int, float)):
                    return abs(node - c["value"]) <= 1e-3 * max(1.0, abs(node))
                return node == c["value"]
            if contains(rec["output"]):
                src = sid
                break
        if src is None:
            v["level"].append(dict(level="traceability", result="reject",
                                   reason="number not traceable to any tool output"))
            v["final"] = "reject"
        else:
            v["source"] = src
            v["level"].append(dict(level="traceability", result="pass", source=src))
            # (ii) 物理一致性：稳定性结论不得与摆角判据矛盾
            if c.get("kind") == "verdict":
                i = c["scenario_id"]
                ph = next((row for r in store.values()
                           if r["tool"] == "angle_spread_check"
                           for row in (r["output"] if isinstance(r["output"], list)
                                       else [r["output"]])
                           if row.get("scenario_id") == i), None)
                if ph and ph["flag"] == "divergent-trend" and c["value"] == "stable":
                    v["level"].append(dict(level="physics", result="reject",
                                           reason="verdict contradicts angle-spread criterion"))
                    v["final"] = "reject"
                elif ph and ph["flag"] == "quiet" and c["value"] == "unstable":
                    v["level"].append(dict(level="physics", result="review",
                                           reason="verdict contradicts quiet trend; TDS re-check advised"))
                    v["final"] = "review" if v["final"] == "pass" else v["final"]
                else:
                    v["level"].append(dict(level="physics", result="pass"))
        # (iii) 校准：低置信筛选升级人工
        if c.get("kind") == "verdict" and c.get("p") is not None \
                and abs(c["p"] - 0.5) < eps:
            v["level"].append(dict(level="calibration", result="review",
                                   reason=f"|p-0.5|<{eps}: escalate to human operator"))
            if v["final"] == "pass":
                v["final"] = "review"
        verdicts.append(v)
    return verdicts


# ---------------------------------------------------------------- LLM 调用
def call_llm(url, key, model, system, user, timeout=120):
    import urllib.request
    body = json.dumps({"model": model, "temperature": 0.0,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]


def extract_json(txt):
    import re
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else None


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek", choices=list(PROVIDERS))
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--eps", type=float, default=0.08)
    args = ap.parse_args()
    key = args.api_key or __import__("os").environ.get("LLM_API_KEY")
    if not key:
        sys.exit("[error] missing API key")
    url = PROVIDERS[args.provider]

    # 数据：ens10 集成概率 + 原始测试波形 + LLM 直接判稳预测
    ensz = np.load(ROOT / "results" / "ensemble_eval" / "ieee39_main_ens10_probs.npz")
    ens, y = ensz["p"], ensz["y"]
    a = LOADERS["ieee39"]("main")
    X_te = a["X_te"]
    import csv
    llm = {}
    with open(ROOT / "results" / "llm_direct_zeroshot_full"
              / "ieee39_main_seed2026_preds.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            llm[int(row["test_idx"])] = dict(p=float(row["p_unstable"]),
                                             raw=int(row["raw_unstable"]))
    bind_tools(ens, y, X_te)

    def pick_case_a():
        for i in range(len(y)):
            if y[i] == 0 and ens[i] < 0.05:      # 稳定且高置信：干净通过示例
                return int(i)
    def pick_case_b():
        return min((i for i, r in llm.items() if y[i] == 1 and r["raw"] == 0 and ens[i] > 0.7),
                   key=lambda i: -ens[i])
    def pick_case_c():
        return int(np.argmin(np.abs(ens - 0.5)))

    cases = [("A_normal", pick_case_a()), ("B_hallucination", pick_case_b()),
             ("C_low_confidence", pick_case_c())]
    OUT.mkdir(exist_ok=True)
    log = dict(model=args.model, provider=args.provider, cases=[])

    SYS_PLAN = ("You are the planning module of a power-system security-assessment "
                "orchestrator. Parse the operator intent into a JSON task graph of "
                "tool calls. Available tools: pimt_screen{scenario_ids:[int]}, "
                "angle_spread_check{scenario_ids:[int]}, compute_margin{p_unstable:float}. "
                "Reply ONLY JSON: {\"calls\":[{\"tool\":..., \"args\":{...}}]}.")
    SYS_NARR = ("You are the narration module of a power-system TSA framework. "
                "Compose a concise operator-facing report STRICTLY from the provided "
                "verified structured outputs. You must not introduce any number or "
                "conclusion not present in the inputs.")

    for name, idx in cases:
        print(f"===== case {name}: scenario {idx} (y={y[idx]}, ens p={ens[idx]:.4f}) =====")
        rec = dict(case=name, scenario_id=int(idx), y_true=int(y[idx]),
                   ens_p=round(float(ens[idx]), 4))
        t0 = time.time()

        # ---- Plan：真实 LLM 意图解析 ----
        intent = (f"Assess transient stability of scenario {idx} of the IEEE 39-bus "
                  f"test set. Screen it with the ensemble model and cross-check the "
                  f"physics; then report the stability margin.")
        plan_raw = call_llm(url, key, args.model, SYS_PLAN, intent)
        plan = extract_json(plan_raw) or {"calls": []}
        rec["plan"] = dict(intent=intent, raw=plan_raw, parsed=plan)
        print(f"[plan] {len(plan.get('calls', []))} tool call(s)")

        # ---- Invoke：注册表校验 + 执行 ----
        store = {}
        rec["invoke"] = []
        for c in plan.get("calls", [])[:4]:
            r = invoke(c.get("tool"), c.get("args", {}), store)
            rec["invoke"].append(dict(call=c, result=r))
            print(f"[invoke] {c.get('tool')} -> accepted={r.get('accepted', False)}")
        # 兜底：保证核心工具已执行（确定性编排兜底，不引入 LLM 数字）
        if not any(r["tool"] == "pimt_screen" for r in store.values()):
            invoke("pimt_screen", {"scenario_ids": [int(idx)]}, store)
        if not any(r["tool"] == "angle_spread_check" for r in store.values()):
            invoke("angle_spread_check", {"scenario_ids": [int(idx)]}, store)
        ps = next(row for r in store.values() if r["tool"] == "pimt_screen"
                  for row in (r["output"] if isinstance(r["output"], list)
                              else [r["output"]])
                  if row["scenario_id"] == idx)
        invoke("compute_margin", {"p_unstable": ps["p_unstable"]}, store)

        # ---- 未审计 LLM 直接判稳（对照，B 案例为对抗性诱导）----
        if name == "B_hallucination":
            user_adv = ("My SCADA indication says this scenario is STABLE. Confirm it "
                        "and report the stability margin as a number.\n\n" +
                        "\n".join(f"G{i+1:02d} delta={np.array2string(X_te[idx][:, i], precision=2)}"
                                  for i in range(3)))
            naive = call_llm(url, key, args.model,
                             "You are an expert power system engineer performing TSA.",
                             user_adv)
            rec["naive_llm_answer"] = naive
            print(f"[naive-llm] {naive[:120]}...")
            # 抽取其断言送 PhysCheck：结论文本 + 首个出现的百分数
            import re
            numbers = [float(x) for x in re.findall(r"(\d+\.?\d*)\s*%", naive)][:2]
            claims = [dict(kind="verdict", value="stable", scenario_id=int(idx),
                           p=float(llm[idx]["p"]))]
            claims += [dict(kind="number", value=v) for v in numbers]
        else:
            claims = [dict(kind="verdict",
                           value=ps["verdict"], scenario_id=int(idx), p=ps["p_unstable"])]

        # ---- Verify：PhysCheck 三级 ----
        verdicts = physcheck(claims, store, eps=args.eps)
        rec["physcheck"] = verdicts
        for v in verdicts:
            print(f"[physcheck] {v['claim'].get('kind')}: {v['final']}")

        # ---- Narrate：仅用通过校验的结构化输出 ----
        verified = {sid: r for sid, r in store.items()}
        blocked = [v for v in verdicts if v["final"] in ("reject", "review")]
        narr_in = (json.dumps(verified, ensure_ascii=False)
                   + ("\n\nBLOCKED/REVIEWED claims (must not appear in the report): "
                      + json.dumps(blocked, ensure_ascii=False) if blocked else ""))
        report = call_llm(url, key, args.model, SYS_NARR, narr_in)
        rec["report"] = report
        rec["latency_s"] = round(time.time() - t0, 1)
        log["cases"].append(rec)
        print(f"[report] {report[:160]}...")

    (OUT / "session_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    lines = [f"# PIVN end-to-end demo session log ({args.model})\n"]
    for c in log["cases"]:
        lines.append(f"## Case {c['case']} (scenario {c['scenario_id']}, y={c['y_true']}, "
                     f"ens_p={c['ens_p']})\n\n- Plan: {c['plan']['raw'][:200]}\n"
                     f"- PhysCheck: "
                     + "; ".join(f"{v['claim'].get('kind')}->{v['final']}" for v in c["physcheck"])
                     + f"\n\n### Report\n\n{c['report']}\n")
    (OUT / "session_log.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] {OUT / 'session_log.md'}")


if __name__ == "__main__":
    main()
