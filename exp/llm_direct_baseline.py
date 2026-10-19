# -*- coding: utf-8 -*-
"""通用 LLM 直接判稳基线：原始波形文本化 → LLM 零样本/少样本二分类。

目的：回应 ESWA/TPS 审稿人必问的"为什么不直接问 LLM"——用与 PIMT-Net 相同的
测试划分做严格对比。输出 JSON 与 train.py 结果同 schema，可进 aggregate/export。

用法（OpenAI 兼容接口任选其一，API key 放环境变量 LLM_API_KEY）：
  # 先看文本化效果（不发请求）
  python exp/llm_direct_baseline.py --dry-run

  # DeepSeek 零样本 100 例
  python exp/llm_direct_baseline.py --provider deepseek --model deepseek-chat --n 100

  # GLM 少样本（4-shot）
  python exp/llm_direct_baseline.py --provider zhipu --model glm-4-plus --mode fewshot

  # Qwen / OpenAI / 任意兼容端点
  python exp/llm_direct_baseline.py --provider dashscope --model qwen-max
  python exp/llm_direct_baseline.py --provider openai --model gpt-4o
  python exp/llm_direct_baseline.py --base-url https://xx/v1 --model xx

特性：请求缓存（断点续跑不重复计费）、解析失败重试、分层抽样固定种子、
逐样本预测 CSV 落盘供错误分析。
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.dataset import LOADERS
from common.metrics import cls_metrics

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

PROVIDERS = {  # OpenAI 兼容 chat/completions 端点
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
}

SYS_PROMPT = (
    "You are an expert power system engineer performing transient stability "
    "assessment (TSA). You are given post-fault measurements of an IEEE 39-bus "
    "system over a short observation window, for each generator: rotor angle "
    "delta (rad), rotor speed omega (pu, 1.0 = nominal), and electrical power "
    "Pe (pu). Judge whether the multi-machine system will lose synchronism "
    "(transient instability), i.e., rotor-angle divergence / sustained "
    "oscillation growth among generators. Reply ONLY a JSON object: "
    '{"unstable": 0 or 1, "confidence": <float 0-1>}. '
    "unstable=1 means the system loses synchronism."
)


def fmt_seq(v, dec=4):
    return "[" + ",".join(f"{x:.{dec}f}" for x in v) + "]"


def to_text(X, n_mach, step):
    """X: (T, 3*n_mach)，通道按量分块 [δ|ω|Pe]。step=1 每步全保留，>2 抽稀。"""
    T = X.shape[0]
    idx = list(range(0, T, step))
    if idx[-1] != T - 1:
        idx.append(T - 1)                      # 保留末点（趋势关键）
    n_m = X.shape[1] // 3
    d, w, p = X[:, :n_m], X[:, n_m:2 * n_m], X[:, 2 * n_m:]
    lines = [f"observation timesteps: {len(idx)} of {T} (window end included)"]
    for i in range(n_m):
        lines.append(f"G{i + 1:02d} delta={fmt_seq(d[idx, i])} "
                     f"omega={fmt_seq(w[idx, i])} Pe={fmt_seq(p[idx, i])}")
    return "\n".join(lines)


def build_prompt(x, mode, examples=None, n_mach=10, step=2):
    parts = []
    if mode == "fewshot" and examples:
        parts.append("Labeled examples from the same system (for calibration):")
        for xe, ye in examples:
            parts.append(to_text(xe, n_mach, step)
                         + f'\n=> {{"unstable": {ye}, "confidence": 1.0}}\n')
        parts.append("Now assess the following case.")
    parts.append(to_text(x, n_mach, step))
    parts.append('Answer with JSON only: {"unstable": 0 or 1, "confidence": 0-1}')
    return "\n\n".join(parts)


def call_llm(url, key, model, prompt, max_retry=3, timeout=90):
    import urllib.error
    import urllib.request
    body = json.dumps({"model": model, "temperature": 0.0,
                       "messages": [{"role": "system", "content": SYS_PROMPT},
                                    {"role": "user", "content": prompt}]}).encode()
    for att in range(max_retry):
        try:
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())["choices"][0]["message"]["content"], None
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:200]
            if e.code in (429, 500, 502, 503) and att < max_retry - 1:
                time.sleep(2 ** att * 2)
                continue
            return None, f"HTTP {e.code}: {detail}"
        except Exception as e:                              # noqa: BLE001
            if att == max_retry - 1:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(2 ** att * 2)
    return None, "unreachable"


def parse_answer(text):
    """从回复中稳健提取 (unstable, confidence)；失败返回 None。"""
    if not text:
        return None
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if m:
        try:
            j = json.loads(m.group())
            u = int(bool(j.get("unstable", j.get("label", 0))))
            c = float(j.get("confidence", 0.5))
            return u, min(max(c, 0.0), 1.0)
        except (ValueError, TypeError):
            pass
    t = text.lower()
    if re.search(r"\b(unstable|instab|lose|1)\b", t) and not re.search(r"\bstable\b", t):
        return 1, 0.5
    if "stable" in t:
        return 0, 0.5
    return None


def stratified_idx(y, n, seed):
    rng = np.random.default_rng(seed)
    pick = []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        k = round(n * len(idx) / len(y))               # 按测试集自然占比分层
        pick += list(rng.choice(idx, min(k, len(idx)), replace=False))
    rng.shuffle(pick)
    return np.array(pick, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek", choices=list(PROVIDERS))
    ap.add_argument("--base-url", default=None, help="自定义 OpenAI 兼容端点")
    ap.add_argument("--api-key", default=None, help="默认读环境变量 LLM_API_KEY")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--dataset", default="ieee39", choices=sorted(LOADERS))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n", type=int, default=100, help="测试样本数（分层抽样）")
    ap.add_argument("--mode", default="zeroshot", choices=["zeroshot", "fewshot"])
    ap.add_argument("--shots", type=int, default=4, help="few-shot 示例数（一半稳定一半失稳）")
    ap.add_argument("--step", type=int, default=2, help="时间抽稀步长（2=隔步采样）")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--name", default=None, help="结果目录名，默认 llm_direct_<mode>")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="只构建并打印 1 个 prompt")
    args = ap.parse_args()

    a = LOADERS[args.dataset](args.tag)
    n_mach = a["X"].shape[2] // 3
    X_te, y_te = a["X_te"].astype(np.float64), a["y"][a["te"]]
    idx = stratified_idx(y_te, args.n, args.seed)

    # few-shot 示例取自训练集（不泄漏），固定一半稳定一半失稳
    examples = None
    if args.mode == "fewshot":
        rng = np.random.default_rng(args.seed)
        y_all, X_all = a["y"][a["tr"]], a["X"].astype(np.float64)[a["tr"]]
        examples = [(X_all[rng.choice(np.where(y_all == c)[0])], c)
                    for c in (0, 1) for _ in range(args.shots // 2)]

    name = args.name or f"llm_direct_{args.mode}"
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f".cache_{args.model.replace('/', '_')}_{args.dataset}_{args.tag}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    prompt0 = build_prompt(X_te[idx[0]], args.mode, examples, n_mach, args.step)
    if args.dry_run:
        print(f"[dry-run] 测试集 n={len(y_te)} 抽样 n={len(idx)} "
              f"(失稳 {int(y_te[idx].sum())})，n_mach={n_mach}")
        print(f"prompt 字符数≈{len(prompt0)}（约 {len(prompt0) // 3} tokens）")
        print("-" * 60 + "\n" + prompt0[:3000])
        return

    key = args.api_key or os.environ.get("LLM_API_KEY")
    if not key:
        sys.exit("[error] 缺少 API key：--api-key 或环境变量 LLM_API_KEY")
    url = args.base_url or PROVIDERS[args.provider]

    from concurrent.futures import ThreadPoolExecutor
    preds, errs, cache_hits = {}, 0, 0
    t0 = time.time()

    def work(k):
        nonlocal cache_hits, errs
        i = idx[k]
        prompt = build_prompt(X_te[i], args.mode, examples, n_mach, args.step)
        ck = hashlib.md5(f"{args.model}|{prompt}".encode()).hexdigest()
        if ck in cache:
            cache_hits += 1
            return k, cache[ck]
        text, err = call_llm(url, key, args.model, prompt)
        if err is not None:
            errs += 1
            print(f"  [warn] sample#{k} 请求失败: {err}")
            return k, None
        ans = parse_answer(text)
        if ans is None:
            errs += 1
            print(f"  [warn] sample#{k} 解析失败: {str(text)[:120]}")
            return k, None
        cache[ck] = ans
        return k, ans

    with ThreadPoolExecutor(args.concurrency) as ex:
        for k, ans in ex.map(work, range(len(idx))):
            preds[k] = ans
            if (k + 1) % 10 == 0:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                done = sum(1 for v in preds.values() if v)
                print(f"  [{k + 1}/{len(idx)}] done={done} err={errs} cache_hit={cache_hits}")

    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    ok = [k for k, v in preds.items() if v is not None]
    y_s, p_s = y_te[idx[ok]], np.array([preds[k][1] if preds[k][0] == 1
                                        else 1.0 - preds[k][1] for k in ok])
    # 语义对齐：confidence 是"对自身判断的确信度"→ 转为 P(unstable)
    m = cls_metrics(y_s, p_s)
    m = {k2: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
         for k2, v in m.items()}
    result = dict(model=name, dataset=args.dataset, tag=args.tag, seed=args.seed,
                  params=None, fit_s=round(time.time() - t0, 1), **m,
                  llm_model=args.model, mode=args.mode, n_eval=len(ok),
                  n_err=errs, cache_hits=cache_hits,
                  base_rate=round(float(y_te[idx].mean()), 4))
    out_json = out_dir / f"{args.dataset}_{args.tag}_seed{args.seed}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    import csv
    with open(out_dir / f"{args.dataset}_{args.tag}_seed{args.seed}_preds.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_idx", "y_true", "p_unstable", "raw_unstable", "confidence"])
        for k in ok:
            u, c = preds[k]
            w.writerow([int(idx[k]), int(y_te[idx[k]]), round(float(p_s[ok.index(k)]), 4), u, c])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[ok] {out_json}")


if __name__ == "__main__":
    main()
