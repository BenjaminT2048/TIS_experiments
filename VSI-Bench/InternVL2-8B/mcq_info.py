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
from transformers import AutoModel, AutoTokenizer


# ----------------------------
# Prompting for rel-direction (INFO-ONLY)
# ----------------------------
# REL_DIR_SYSTEM_INSTR = (
#     "Rules:\n"
#     "1) Read the question and options (A/B/C/D).\n"
#     "2) Assume a local coordinate system exactly as described in the question.\n"
#     "3) Do NOT show any reasoning or explanation.\n"
#     "4) On the LAST line, output exactly one letter in [A,B,C,D] with no extra text."
# )
REL_DIR_SYSTEM_INSTR = (
    "Rules:\n"
    "1) Read the question and options (A/B/C/D).\n"
    "2) Assume a local coordinate system exactly as described in the question.\n"
    "3) You are encourage to output your thinking process.\n"
    "4) On the LAST line, output exactly one letter in [A,B,C,D] with no extra text."
)

def build_rel_dir_prompt_info_only(
    question: str,
    options_text: str,
    obj2d_text: Optional[str] = None
) -> str:
    """
    Build a prompt that contains exactly ONE <image> placeholder (blank) and the JSON block.
    The image is explicitly declared as blank; the model must rely ONLY on JSON.
    """

    obj2d_block = ""
    if obj2d_text:
        obj2d_block = (
            "The JSON contains view-invariant 2D positions (in centimeters) of objects in this scene. For each label, the list contains all instances (length = count; each [x,y] is one instance).\n"
            "<OBJECT_2D_INFO_JSON>\n"
            f"{obj2d_text}\n"
            "</OBJECT_2D_INFO_JSON>\n"
        )

    prompt = (
        f"{REL_DIR_SYSTEM_INSTR}\n"
        f"{obj2d_block}\n"
        f"Question: {question.strip()}\n"
        f"Options: {options_text.strip()}\n"
        f"You are encourage to output your thinking process. Output exactly one of A, B, C, or D in the last line."
    )
    return prompt


CHOICE_RE = re.compile(r'\b([ABCD])\b', re.IGNORECASE)

def extract_choice_letter(text: str) -> Optional[str]:
    """
    Try to get the final single-letter choice from model output.
    Prefer the last line if it contains A/B/C/D; fallback to last match.
    """
    if not isinstance(text, str):
        return None
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        m = CHOICE_RE.search(lines[-1])
        if m:
            return m.group(1).upper()
    ms = list(CHOICE_RE.finditer(text))
    return ms[-1].group(1).upper() if ms else None


# ----------------------------
# Model wrapper
# ----------------------------
def load_internvl2(model_path: str, device: str = "cuda"):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    # sanity check for generate (common pitfall with transformers versions)
    if not hasattr(getattr(model, "language_model", model), "generate"):
        raise RuntimeError("Underlying language model has no .generate(); "
                           "pin transformers to a compatible version (e.g., 4.37.2 for InternVL2).")
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
            return [round(x, nd), round(y, nd)]
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
        filtered_labels = [lbl for lbl, c in counts.items() if c <= 10]
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
def answer_rel_direction_info_only(
    model,
    tokenizer,
    question: str,
    options_text: str,
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

    # 1 blank image as placeholder
    pixel_values = torch.zeros((1, 3, input_size, input_size), dtype=torch.bfloat16, device=device)
    num_patches_list = [1]  # exactly one placeholder image

    prompt = build_rel_dir_prompt_info_only(
        question=question,
        options_text=options_text,
        obj2d_text=obj2d_text
    )

    response = model.chat(
        tokenizer,
        pixel_values,
        prompt,
        gen_cfg,
        num_patches_list=num_patches_list,
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
    ap = argparse.ArgumentParser(description="Run InternVL2 INFO-ONLY on VSI-Bench object_rel_direction/distance (arkitscenes).")
    ap.add_argument("--csv", type=str, default="VSI-Bench/test-00000-of-00001.csv", help="Path to VSI-Bench CSV (converted from parquet).")
    ap.add_argument("--model_path", type=str, default="OpenGVLab/InternVL2-8B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--input_size", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--qtype", type=str, required=True)
    ap.add_argument("--obj2d_dir", type=str, default="ARKitScenes/2D_annotation",
                    help="Directory containing per-scene JSON with view-invariant 2D positions; "
                         "filenames start with scene_name (e.g., '41069025.json' or '41069025_annotation.json').")
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
    model, tokenizer = load_internvl2(args.model_path, device=device)

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
            options_text = str(row.get("options", ""))
            gt = str(row.get("ground_truth", "")).strip()


            out = answer_rel_direction_info_only(
                model, tokenizer,
                question=q,
                options_text=options_text,
                input_size=args.input_size,
                gen_cfg=dict(max_new_tokens=1024, do_sample=False, temperature=0.0),
                device=device,
                obj2d_text=obj2d_text
            )
            pred = out["predicted_letter"]
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
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    if n_total > 0:
        acc = n_correct / n_total
        print(f"Done. Evaluated {n_total} with GT letters. Accuracy: {n_correct}/{n_total} = {acc:.3f}")
    else:
        print("Done. No samples with letter GT were evaluated.")

    from datetime import datetime
    import math

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
            "EM": {
                "n_total": n_total,
                "n_correct": n_correct,
                "accuracy": _clean_float(n_correct / n_total if n_total else None),
            }
        },
        "args": vars(args)  # 记录本次运行配置，便于复现
    }
        
    with open(args.out, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


# python /root/TIS/VSI-Bench/InternVL2-8B/mcq_info.py --out /root/TIS/VSI-Bench/results/object_rel_direction/r1.json --qtype object_rel_direction
# python /root/TIS/VSI-Bench/InternVL2-8B/mcq_info.py --out /root/TIS/VSI-Bench/results/object_rel_distance/pure_info.json --qtype object_rel_distance



# python /root/TIS/VSI-Bench/InternVL2-8B/mcq_info.py --out /root/TIS/VSI-Bench/results/object_rel_distance/pure_info_cot.json --qtype object_rel_distance
