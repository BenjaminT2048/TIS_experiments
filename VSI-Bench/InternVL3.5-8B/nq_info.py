import os
import re
import json
import argparse
from typing import Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

from mra_eval import (
    parse_number, accumulate_mra_stats, evaluate_numeric_pair
)

# ----------------------------
# Prompting for rel-direction (INFO-ONLY)
# ----------------------------
REL_DIR_SYSTEM_INSTR = (
    "Rules:\n"
    "1) Read the question.\n"
    "2) Assume a local coordinate system exactly as described in the question.\n"
    "3) Do NOT show any reasoning or explanation.\n"
    "4) On the LAST line, output exactly one number with no extra text."
)

# REL_DIR_SYSTEM_INSTR = (
#     "Rules:\n"
#     "1) Read the question.\n"
#     "2) Assume a local coordinate system exactly as described in the question.\n"
#     "3) You are encourage to output your thinking process.\n"
#     "4) On the LAST line, output exactly one number with no extra text."
# )


def build_prompt(
    question: str,
    obj2d_text: Optional[str] = None
) -> str:
    """
    Build a prompt that contains exactly ONE <image> placeholder (blank) and the JSON block.
    The image is explicitly declared as blank; the model must rely ONLY on JSON.
    """

    obj2d_block = ""
    if obj2d_text:
        obj2d_block = (
            "The JSON contains view-invariant 2D positions (in meters) of objects in this scene. For each label, the list contains all instances (length = count; each [x,y] is one instance).\n"
            "<OBJECT_2D_INFO_JSON>\n"
            f"{obj2d_text}\n"
            "</OBJECT_2D_INFO_JSON>\n"
        )

    # prompt = (
    #     f"{REL_DIR_SYSTEM_INSTR}\n"
    #     f"{obj2d_block}\n"
    #     f"Question: {question.strip()}\n"
    #     f"You are encourage to output your thinking process. Output exactly one number as the answer in the last line."
    # )
    prompt = (
        f"{REL_DIR_SYSTEM_INSTR}\n"
        f"{obj2d_block}\n"
        f"Question: {question.strip()}\n"
        f"Output exactly one number as the answer in the last line."
    )
    return prompt


NUMBER_RE = re.compile(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?')

def extract_choice_letter(text: str) -> Optional[str]:
    """
    Try to get the final numeric answer from model output.
    Accepts integers, decimals, or scientific notation.
    Prefer the last non-empty line; fallback to the last match in the whole text.
    Returns the matched number as a string.
    """
    if not isinstance(text, str):
        return None
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        m = NUMBER_RE.search(lines[-1])
        if m:
            return m.group(0)
    ms = list(NUMBER_RE.finditer(text))
    return ms[-1].group(0) if ms else None



# ----------------------------
# Model wrapper
# ----------------------------
def load_model(model_path: str, device: str = "cuda"):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


# ----------------------------
# Object-2D JSON loader (robust)
# ----------------------------
def find_scene_obj2d_path(obj2d_dir: Optional[str], scene_name: str) -> Optional[str]:
    """
    Find a JSON file in obj2d_dir whose filename starts with scene_name.
    Prefer exact '{scene_name}.json'; otherwise, the first prefix match.
    """
    if not obj2d_dir or not os.path.isdir(obj2d_dir):
        return None

    exact = os.path.join(obj2d_dir, f"{scene_name}.json")
    if os.path.isfile(exact):
        return exact

    for fname in os.listdir(obj2d_dir):
        if fname.startswith(scene_name) and fname.endswith(".json"):
            return os.path.join(obj2d_dir, fname)
    return None


def load_obj2d_as_prompt_text(
    json_path: str,
    max_chars: int = 8000,
    decimals: int = 1,
) -> Optional[str]:
    """
    读取形如 [{"label": str, "centroid_xy": [x, y]}, ...] 的 JSON。
    规则：若某 label 在文件中出现次数 > 2，则整类删除；其余 (<=2) 全保留并四舍五入。
    输出紧凑 JSON：
      {"coord_system":"view-invariant 2D (x,y)","coords_by_label":{"bed":[[x,y],...], ...}}
    """
    import json

    def _round_xy(xy, nd):
        try:
            x, y = float(xy[0]), float(xy[1])
            return [round(x, nd)*0.01, round(y, nd)*0.01]
        except Exception:
            return None

    def _build_text(data, nd, keep_top_labels: Optional[int] = None):
        counts = {}
        grouped = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                lbl = item.get("label")
                xy = item.get("centroid_xy")
                if lbl is None or xy is None or not isinstance(xy, (list, tuple)) or len(xy) < 2:
                    continue
                counts[lbl] = counts.get(lbl, 0) + 1
                grouped.setdefault(lbl, []).append(xy)
        else:
            return None

        # 保留 <=2；删除 >2
        filtered_labels = [lbl for lbl, _ in counts.items()]
        filtered_labels.sort()

        if keep_top_labels is not None and keep_top_labels < len(filtered_labels):
            filtered_labels = filtered_labels[:keep_top_labels]

        coords_by_label = {}
        for lbl in filtered_labels:
            pts = grouped.get(lbl, [])
            rounded_pts = []
            for xy in pts:
                r = _round_xy(xy, nd)
                if r is not None:
                    rounded_pts.append(r)
            if rounded_pts:
                coords_by_label[str(lbl)] = rounded_pts

        obj = {
            "coord_system": "view-invariant 2D (x,y)",
            "coords_by_label": coords_by_label
        }
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    if not json_path or not os.path.isfile(json_path):
        return None

    # 尝试解析 → 紧凑化
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # 解析失败兜底：直接原文截断
        try:
            raw = open(json_path, "r", encoding="utf-8", errors="ignore").read().strip()
            return raw[:max_chars - 20] + "...(truncated)" if len(raw) > max_chars else raw
        except Exception:
            return None

    for nd in [decimals, 0]:
        txt = _build_text(data, nd=nd, keep_top_labels=None)
        if txt is not None and len(txt) <= max_chars:
            return txt
        for keep in [100, 50, 25, 10, 5, 3, 1]:
            txt = _build_text(data, nd=nd, keep_top_labels=keep)
            if txt is not None and len(txt) <= max_chars:
                return txt

    tight = _build_text(data, nd=0, keep_top_labels=1)
    if tight is None:
        try:
            raw_min = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            return raw_min[:max_chars - 20] + "...(truncated)" if len(raw_min) > max_chars else raw_min
        except Exception:
            return None
    return tight[:max_chars - 20] + "...(truncated)"


# ----------------------------
# INFO-ONLY answering (no video)
# ----------------------------
@torch.no_grad()
def answer(
    model,
    tokenizer,
    question: str,
    input_size: int = 448,
    gen_cfg: Optional[Dict] = None,
    device: str = "cuda",
    obj2d_text: Optional[str] = None
) -> Dict:
    """
    不读取视频；提供一个全零的占位图像 (1x3xH x W)，并强制在 prompt 中声明“只用 JSON 推理”。
    """
    if gen_cfg is None:
        gen_cfg = dict(max_new_tokens=1024, do_sample=False, temperature=0.0)


    prompt = build_prompt(
        question=question,
        obj2d_text=obj2d_text
    )

    response = model.chat(
        tokenizer,
        None,
        prompt,
        gen_cfg,
    )
    pred_letter = extract_choice_letter(response)
    return {
        "raw_response": response,
        "predicted_letter": pred_letter,
        "prompt": prompt
    }


# ----------------------------
# Main runner
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("--csv", type=str, default="VSI-Bench/test-00000-of-00001.csv", help="Path to VSI-Bench CSV (converted from parquet).")
    ap.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--input_size", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--qtype", type=str, required=True)
    ap.add_argument("--obj2d_dir", type=str, default="ARKitScenes/2D_annotation")
    ap.add_argument("--obj2d_max_chars", type=int, default=8000,
                    help="Max characters of object-2D JSON to inject into prompt (will be compacted/truncated safely).")
    args = ap.parse_args()

    # load data
    df = pd.read_csv(args.csv)
    qtype = df["question_type"].astype(str)
    # 同时支持 direction / distance（可按需改成只 direction）
    df = df[(df["dataset"] == "arkitscenes") & (qtype.str.startswith((args.qtype)))]

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    print(f"Selected {len(df)} samples from CSV.")

    # load model
    device = args.device
    model, tokenizer = load_model(args.model_path, device=device)

    # prepare output (resume-safe)
    seen_ids = set()
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen_ids.add(rec.get("id"))
                except Exception:
                    pass
        print(f"[resume] found {len(seen_ids)} existing results; will skip those ids.")

    n_correct = 0
    n_total = 0
    relerrs = []

    with open(args.out, "a", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sid = row.get("id")
            if sid in seen_ids:
                continue

            scene_name = str(row["scene_name"]).strip()

            # locate & load per-scene object-2D JSON
            obj2d_path = find_scene_obj2d_path(args.obj2d_dir, scene_name) if args.obj2d_dir else None
            obj2d_text = load_obj2d_as_prompt_text(obj2d_path, max_chars=args.obj2d_max_chars) if obj2d_path else None
            # 兜底：文件存在但处理失败时，注入原文
            if obj2d_path and not obj2d_text:
                try:
                    obj2d_text = open(obj2d_path, "r", encoding="utf-8", errors="ignore").read()[:args.obj2d_max_chars]
                except Exception:
                    obj2d_text = None

            q = str(row["question"])
            gt = str(row.get("ground_truth", "")).strip()


            out = answer(
                model, tokenizer,
                question=q,
                input_size=args.input_size,
                gen_cfg=dict(max_new_tokens=1024, do_sample=False, temperature=0.0),
                device=device,
                obj2d_text=obj2d_text
            )
            pred = out["predicted_letter"]

            
            pred_num, gt_num, relerr, sample_mra = evaluate_numeric_pair(pred, gt)
            if relerr is not None:
                relerrs.append(relerr)

            record = {
                "id": sid,
                "dataset": row["dataset"],
                "scene_name": scene_name,
                "question_type": row["question_type"],
                "question": q,
                "ground_truth": gt,
                "predicted_answer": pred,
                "relative_error": relerr,
                "sample_mra": sample_mra,
                "prompt": out["prompt"],
                "raw_response": out["raw_response"],
                "obj2d_path": obj2d_path,
                "obj2d_in_prompt": bool(obj2d_text),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = accumulate_mra_stats(relerrs)
    if stats.n_valid > 0:
        print(f"[NA] MRA on {stats.n_valid} numeric samples: {stats.mean_mra:.4f}")
        per_th = ", ".join([f"θ={t:.2f}:{a:.3f}" for t, a in stats.per_threshold_acc.items()])
        print(f"[NA] Per-threshold accuracy -> {per_th}")
    else:
        print("[NA] No numeric samples (or unparsable); MRA not computed.")


    from datetime import datetime
    import math
    from mra_eval import MRA_THRESHOLDS  # 如果你把阈值放在独立文件

    def _clean_float(x):
        if x is None:
            return None
        try:
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return None
            return xf
        except Exception:
            return None
    
    summary_rec = {
        "type": "summary",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "file": args.out,
        "dataset": "arkitscenes",
        "qtypes": sorted(df["question_type"].astype(str).unique().tolist()),
        "metrics": {
            "mc": {
                "n_total": n_total,
                "n_correct": n_correct,
                "accuracy": _clean_float(n_correct / n_total if n_total else None),
            },
            "na": {
                "n_valid": int(len(relerrs)),
                "mean_mra": _clean_float(stats.mean_mra if 'stats' in locals() else None),
                "per_threshold_acc": {f"{t:.2f}": _clean_float(a) for t, a in (stats.per_threshold_acc.items() if 'stats' in locals() else {})},
                "mean_relative_error": _clean_float(float(np.mean(relerrs)) if len(relerrs) else None),
            },
        },
        "mra_thresholds": [float(t) for t in MRA_THRESHOLDS],
        "args": vars(args),  # 记录本次运行配置，便于复现
    }

    summary_rec = {
        "type": "summary",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "file": args.out,
        "dataset": "arkitscenes",
        "qtypes": sorted(df["question_type"].astype(str).unique().tolist()),
        "metrics": {
            "nq": {
                "n_valid": int(len(relerrs)),
                "mean_mra": _clean_float(stats.mean_mra if 'stats' in locals() else None),
                "per_threshold_acc": {f"{t:.2f}": _clean_float(a) for t, a in (stats.per_threshold_acc.items() if 'stats' in locals() else {})},
                "mean_relative_error": _clean_float(float(np.mean(relerrs)) if len(relerrs) else None),
            },
        },
        "mra_thresholds": [float(t) for t in MRA_THRESHOLDS],
        "args": vars(args),  # 记录本次运行配置，便于复现
    }

    with open(args.out, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")
        
    


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


# python VSI-Bench/InternVL3.5-8B/nq_info.py --out VSI-Bench/InternVL3.5-8B/results/object_counting/pure_info.json --qtype object_counting