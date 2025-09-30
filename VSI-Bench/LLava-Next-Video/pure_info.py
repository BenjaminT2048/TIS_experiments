#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
# >>> 仅保留 transformers 中的 LLaVA 组件
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

from mra_eval import (
    parse_number, accumulate_mra_stats, evaluate_numeric_pair
)


SYSTEM_INSTR_mcq = (
    "Rules:\n"
    "1) Read the question and options (A/B/C/D).\n"
    "2) Assume a local coordinate system exactly as described in the question.\n"
    "3) Do NOT show any reasoning or explanation.\n"
    "4) On the LAST line, output exactly one letter in [A,B,C,D] with no extra text."
)

SYSTEM_INSTR_mcq_cot = (
    "Rules:\n"
    "1) Read the question and options (A/B/C/D).\n"
    "2) Assume a local coordinate system exactly as described in the question.\n"
    "3) You are encourage to output your thinking process.\n"
    "4) On the LAST line, output exactly one letter in [A,B,C,D] with no extra text."
)

SYSTEM_INSTR_nq = (
    "Rules:\n"
    "1) Read the question.\n"
    "2) Assume a local coordinate system exactly as described in the question.\n"
    "3) Do NOT show any reasoning or explanation.\n"
    "4) On the LAST line, output exactly one number with no extra text."
)


SYSTEM_INSTR_nq_cot = (
    "Rules:\n"
    "1) Read the question.\n"
    "2) Assume a local coordinate system exactly as described in the question.\n"
    "3) You are encourage to output your thinking process.\n"
    "4) On the LAST line, output exactly one number with no extra text."
)

def build_prompt(
    question: str,
    qtype: str,
    cot: bool,
    options_text: str,
    obj2d_text: Optional[str] = None,
) -> str:
    """
    Build a prompt that contains exactly ONE <image> placeholder (blank) and the JSON block.
    The image is explicitly declared as blank; the model must rely ONLY on JSON.
    """
    if qtype == "mcq":
        SYSTEM_INSTR = SYSTEM_INSTR_mcq_cot if cot else SYSTEM_INSTR_mcq
    elif qtype == "nq":
        SYSTEM_INSTR = SYSTEM_INSTR_nq_cot if cot else SYSTEM_INSTR_nq
    else:
        SYSTEM_INSTR = SYSTEM_INSTR_mcq  # 兜底

    obj2d_block = ""
    if obj2d_text:
        obj2d_block = (
            "The JSON contains view-invariant 2D positions (in meters) of objects in this scene. For each label, the list contains all instances (length = count; each [x,y] is one instance).\n"
            "<OBJECT_2D_INFO_JSON>\n"
            f"{obj2d_text}\n"
            "</OBJECT_2D_INFO_JSON>\n"
        )

    prompt = (
        f"{SYSTEM_INSTR}\n"
        f"{obj2d_block}\n"
        f"Question: {question.strip()}\n"
    )
    if qtype == "mcq":
        prompt += f"Options: {options_text.strip()}\n"

    if cot:
        prompt += "You are encourage to show your thinking process. Output exactly your answer in the last line.\n"
    else:
        prompt += "Do not give any reasoning process. Output exactly your answer in the last line.\n"
    return prompt


CHOICE_RE = re.compile(r'\b([ABCD])\b', re.IGNORECASE)

def extract_choice_letter(text: str) -> Optional[str]:
    """
    Try to get the final single-letter choice from model output.
    Prefer the last line if it contains A/B/C/D; fallback to last match.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        m = CHOICE_RE.search(lines[-1])
        if m:
            return m.group(1).upper()
    ms = list(CHOICE_RE.finditer(text))
    return ms[-1].upper() if ms else None  # .upper() on whole match works since it's one letter

# 允许：-3, +3, 3, 3., .5, 3.14, 1e-3, -2.0E+5 等
NUMBER_RE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')

def extract_numerical_answer(text: str) -> Optional[str]:
    """
    从模型输出中抓取最后一行中的数字答案（可能是整数/小数/科学计数法）。
    只看最后一行：若最后一行没有数字，返回 None。
    """
    if not isinstance(text, str):
        return None
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    matches = list(NUMBER_RE.finditer(last))
    return matches[-1].group(0) if matches else None


# ----------------------------
# Model wrapper (LLaVA-NeXT-Video)
# ----------------------------
def load_model(model_path: str, device: str = "cuda"):
    """
    加载 LLaVA-NeXT-Video 与对应 processor；本脚本“纯信息不看视频”，因此后续仅做文本生成。
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float16 if dev.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(dev)
    processor = LlavaNextVideoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor, dev


# ----------------------------
# INFO-ONLY answering (no video)
# ----------------------------
@torch.no_grad()
def answer(
    model,
    processor,
    device,
    question: str,
    qtype: str,
    cot: bool,
    options_text: str,
    input_size: int = 448,                 # 保留签名以兼容；本实现不使用
    gen_cfg: Optional[Dict] = None,
    obj2d_text: Optional[str] = None
) -> Dict:
    """
    不读取视频；通过 chat template 构建仅文本的对话输入，让模型只基于 JSON 与题干作答。
    """
    if gen_cfg is None:
        gen_cfg = dict(max_new_tokens=1024, do_sample=False, temperature=0.0)

    prompt = build_prompt(
        question=question,
        obj2d_text=obj2d_text,
        qtype=qtype,
        cot=cot,
        options_text=options_text,
    )

    # 用 LLaVA 的 chat template 构造仅文本会话
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    templated = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # 仅文本 → tokenizer 打包
    inputs = processor(text=templated, return_tensors="pt").to(device)

    # 生成
    output_ids = model.generate(
        **inputs,
        max_new_tokens=gen_cfg.get("max_new_tokens", 1024),
        do_sample=gen_cfg.get("do_sample", False),
        temperature=gen_cfg.get("temperature", 0.0),
    )

    # 解码：去掉提示长度，只保留新生成段
    input_len = inputs["input_ids"].shape[-1]
    gen_only = output_ids[0][input_len:]
    # processor 暴露 tokenizer
    text_out = processor.tokenizer.decode(gen_only, skip_special_tokens=True)

    return {
        "raw_response": text_out,
        "prompt": prompt
    }


# ----------------------------
# Main runner
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("--csv", type=str, default="VSI-Bench/test-00000-of-00001.csv", help="Path to VSI-Bench CSV (converted from parquet).")
    # 默认模型改为 LLaVA-NeXT-Video
    ap.add_argument("--model_path", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--input_size", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    ap.add_argument("--obj2d_max_chars", type=int, default=8000, help="Max characters of object-2D JSON to inject into prompt (will be compacted/truncated safely).")
    ap.add_argument("--cot", type=bool, default=False)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--qtype", type=str, required=True, choices=["object_rel_direction", "object_rel_distance", "obj_appearance_order", "route_planning",
                                                                 "object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"])
    ap.add_argument("--obj2d_dir", type=str, default="ARKitScenes/2D_annotation")

    args = ap.parse_args()

    # load data
    df = pd.read_csv(args.csv)
    qtype_series = df["question_type"].astype(str)
    df = df[(df["dataset"] == "arkitscenes") & (qtype_series.str.startswith((args.qtype)))]

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    print(f"Selected {len(df)} samples from CSV.")

    # load model (LLaVA)
    device = args.device
    model, processor, dev = load_model(args.model_path, device=device)

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
        mcq = args.qtype in ["object_rel_direction", "object_rel_distance", "obj_appearance_order", "route_planning"]
        nq = args.qtype in ["object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"]
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sid = row.get("id")
            if sid in seen_ids:
                continue

            scene_name = str(row["scene_name"]).strip()

            # locate & load per-scene object-2D JSON
            obj2d_path = find_scene_obj2d_path(args.obj2d_dir, scene_name) if args.obj2d_dir else None
            obj2d_text = load_obj2d_as_prompt_text(obj2d_path, max_chars=args.obj2d_max_chars) if obj2d_path else None
            if obj2d_path and not obj2d_text:
                try:
                    obj2d_text = open(obj2d_path, "r", encoding="utf-8", errors="ignore").read()[:args.obj2d_max_chars]
                except Exception:
                    obj2d_text = None

            q = str(row["question"])
            options_text = str(row.get("options", ""))
            gt = str(row.get("ground_truth", "")).strip()

            out = answer(
                model=model,
                processor=processor,
                device=dev,
                question=q,
                input_size=args.input_size,
                gen_cfg=dict(max_new_tokens=1024, do_sample=False, temperature=0.0),
                obj2d_text=obj2d_text,
                qtype="mcq" if mcq else "nq",
                cot=args.cot,
                options_text=options_text
            )

            if mcq:
                pred = extract_choice_letter(out["raw_response"])
                correct = (pred == gt) if gt in {"A", "B", "C", "D"} and pred is not None else None
                n_total += 1
                if correct is True:
                    n_correct += 1

                record = {
                    "id": sid,
                    "dataset": row["dataset"],
                    "scene_name": scene_name,
                    "question_type": row["question_type"],
                    "question": q,
                    "options": options_text,
                    "ground_truth": gt,
                    "predicted_answer": pred,
                    "correct": correct,
                    "prompt": out["prompt"],
                    "raw_response": out["raw_response"],
                    "obj2d_path": obj2d_path,
                    "obj2d_in_prompt": bool(obj2d_text),
                }
            elif nq:
                pred = extract_numerical_answer(out["raw_response"])
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
                }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    from datetime import datetime
    import math
    from mra_eval import MRA_THRESHOLDS

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

    if args.qtype in ["object_rel_direction", "object_rel_distance", "obj_appearance_order", "route_planning"]:
        summary_rec = {
            "type": "summary",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "file": args.out,
            "dataset": "arkitscenes",
            "qtypes": sorted(df["question_type"].astype(str).unique().tolist()),
            "metrics": {
                "ACC": {
                    "n_total": n_total,
                    "n_correct": n_correct,
                    "accuracy": _clean_float(n_correct / n_total if n_total else None),
                }
            },
            "args": vars(args)
        }
    else:
        stats = accumulate_mra_stats(relerrs)
        if stats.n_valid > 0:
            print(f"[NA] MRA on {stats.n_valid} numeric samples: {stats.mean_mra:.4f}")
            per_th = ", ".join([f"θ={t:.2f}:{a:.3f}" for t, a in stats.per_threshold_acc.items()])
            print(f"[NA] Per-threshold accuracy -> {per_th}")
        else:
            print("[NA] No numeric samples (or unparsable); MRA not computed.")

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
            "args": vars(args),
        }

    with open(args.out, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")


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

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
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


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


# python /workspace/TIS_experiments/VSI-Bench/LLava-Next-Video/pure_info.py --out VSI-Bench/LLava-Next-Video/results/object_rel_direction/pure_info.json --qtype object_rel_direction 

# python VSI-Bench/LLava-Next-Video/pure_info.py --out VSI-Bench/LLava-Next-Video/results/object_rel_distance/pure_info.json --qtype object_rel_distance 


# python VSI-Bench/LLava-Next-Video/pure_info.py --out VSI-Bench/LLava-Next-Video/results/object_abs_distance/pure_info.json --qtype object_abs_distance 


