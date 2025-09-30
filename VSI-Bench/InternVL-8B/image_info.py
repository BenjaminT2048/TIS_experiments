#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from mra_eval import (
    parse_number, accumulate_mra_stats, evaluate_numeric_pair, MRA_THRESHOLDS
)

# ----------------------------
# Prompt system instructions
# ----------------------------
SYSTEM_INSTR_mcq = (
    "Rules:\n"
    "1) You are given a top-down (plan view) image of a room. Use ONLY this image for spatial reasoning.\n"
    "2) Read the question and options (A/B/C/D). Do NOT show any reasoning or explanation.\n"
    "3) On the LAST line, output exactly one letter in [A,B,C,D] with no extra text."
)

SYSTEM_INSTR_mcq_cot = (
    "Rules:\n"
    "1) You are given a top-down (plan view) image of a room. Use ONLY this image for spatial reasoning.\n"
    "2) Read the question and options (A/B/C/D). You are encouraged to show your thinking process.\n"
    "3) On the LAST line, output exactly one letter in [A,B,C,D] with no extra text."
)

SYSTEM_INSTR_nq = (
    "Rules:\n"
    "1) You are given a top-down (plan view) image of a room. Use ONLY this image for spatial reasoning.\n"
    "2) Read the question. Do NOT show any reasoning or explanation.\n"
    "3) On the LAST line, output exactly one number with no extra text."
)

SYSTEM_INSTR_nq_cot = (
    "Rules:\n"
    "1) You are given a top-down (plan view) image of a room. Use ONLY this image for spatial reasoning.\n"
    "2) Read the question. You are encouraged to show your thinking process.\n"
    "3) On the LAST line, output exactly one number with no extra text."
)


def build_prompt(
    question: str,
    qtype: str,
    cot: bool,
    options_text: str,
) -> str:
    """
    Build a prompt that contains exactly ONE <image> placeholder.
    We now rely on a top-down image; no JSON is injected.
    """
    if qtype == "mcq":
        SYSTEM_INSTR = SYSTEM_INSTR_mcq_cot if cot else SYSTEM_INSTR_mcq
    elif qtype == "nq":
        SYSTEM_INSTR = SYSTEM_INSTR_nq_cot if cot else SYSTEM_INSTR_nq
    else:
        SYSTEM_INSTR = SYSTEM_INSTR_mcq  # sensible default

    prompt = (
        f"{SYSTEM_INSTR}\n"
        f"<image>\n"
        f"Question: {question.strip()}\n"
    )
    if qtype == "mcq":
        prompt += f"Options: {options_text.strip()}\n"

    if cot:
        prompt += "You are encourage to show your thinking process. Output exactly your answer in the last line.\n"
    else:
        prompt += "Do not give any reasoning process. Output exactly your answer in the last line.\n"

    return prompt

# ----------------------------
# Output parsers
# ----------------------------
CHOICE_RE = re.compile(r'\b([ABCD])\b', re.IGNORECASE)

def extract_choice_letter(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        m = CHOICE_RE.search(lines[-1])
        if m:
            return m.group(1).upper()
    ms = list(CHOICE_RE.finditer(text))
    return ms[-1].group(1).upper() if ms else None

NUMBER_RE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')

def extract_numerical_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    matches = list(NUMBER_RE.finditer(last))
    return matches[-1].group(0) if matches else None

# ----------------------------
# Model wrapper
# ----------------------------
def load_model(model_path: str, device: str = "cuda"):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto" if device.startswith("cuda") else None,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

# ----------------------------
# InternVL3.5-style image loader (tiling)
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size: int):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    return transform

def _find_closest_aspect_ratio(aspect_ratio: float,
                               target_ratios: List[Tuple[int, int]],
                               width: int, height: int, image_size: int) -> Tuple[int, int]:
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for r in target_ratios:
        target_ar = r[0] / r[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = r
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * r[0] * r[1]:
                best_ratio = r
    return best_ratio

def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    ow, oh = image.size
    aspect_ratio = ow / oh
    target_ratios = sorted(
        set((i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num),
        key=lambda x: x[0] * x[1]
    )
    target_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, ow, oh, image_size)
    target_w = image_size * target_ratio[0]
    target_h = image_size * target_ratio[1]
    blocks   = target_ratio[0] * target_ratio[1]

    resized = image.resize((target_w, target_h))
    processed = []
    grid_w = target_w // image_size
    for i in range(blocks):
        box = (
            (i % grid_w) * image_size,
            (i // grid_w) * image_size,
            ((i % grid_w) + 1) * image_size,
            ((i // grid_w) + 1) * image_size
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed

def load_topdown_image_tensor(image_file: str, input_size=448, max_num=12, device="cuda"):
    img = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    tiles = _dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    # InternVL code commonly casts to bfloat16 on GPU
    if device.startswith("cuda"):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
    else:
        pixel_values = pixel_values.to(torch.bfloat16)
    return pixel_values

# ----------------------------
# Image path resolver
# ----------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

def find_scene_topdown_path(img_dir: Optional[str], scene_name: str) -> Optional[str]:
    """
    Find an image in img_dir whose filename starts with scene_name.
    Prefer exact '{scene_name}.{ext}' over prefix matches.
    """
    if not img_dir or not os.path.isdir(img_dir):
        return None

    # exact match first
    for ext in IMG_EXTS:
        exact = os.path.join(img_dir, f"{scene_name}{ext}")
        if os.path.isfile(exact):
            return exact

    # then prefix matches
    for fname in os.listdir(img_dir):
        low = fname.lower()
        if any(low.endswith(ext) for ext in IMG_EXTS) and fname.startswith(scene_name):
            return os.path.join(img_dir, fname)
    return None

# ----------------------------
# Answering with ONE image (no video)
# ----------------------------
@torch.no_grad()
def answer(
    model,
    tokenizer,
    pixel_values: torch.Tensor,
    question: str,
    qtype: str,
    cot: bool,
    options_text: str,
    gen_cfg: Optional[Dict] = None,
) -> Dict:
    if gen_cfg is None:
        gen_cfg = dict(max_new_tokens=1024, do_sample=False, temperature=0.0)

    prompt = build_prompt(
        question=question,
        qtype=qtype,
        cot=cot,
        options_text=options_text,
    )
    response = model.chat(
        tokenizer,
        pixel_values,  # <-- provide image tiles
        prompt,
        gen_cfg,
    )
    return {
        "raw_response": response,
        "prompt": prompt
    }

# ----------------------------
# Main runner
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate with top-down ROOM IMAGES (no JSON).")
    ap.add_argument("--csv", type=str, default="VSI-Bench/test-00000-of-00001.csv",
                    help="Path to VSI-Bench CSV (converted from parquet).")
    ap.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--input_size", type=int, default=448)
    ap.add_argument("--max_tiles", type=int, default=12, help="Max tiles per image (InternVL tiling).")
    ap.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    ap.add_argument("--cot", type=bool, default=False)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--qtype", type=str, required=True,
                    choices=[
                        "object_rel_direction",
                        "object_rel_distance",
                        "obj_appearance_order",
                        "route_planning",
                        "object_counting",
                        "object_size_estimation",
                        "room_size_estimation",
                        "object_abs_distance"
                    ])
    # NEW: directory containing top-down images per scene
    ap.add_argument("--topdown_dir", type=str, default="info/topdown_images")

    args = ap.parse_args()

    # load data
    df = pd.read_csv(args.csv)
    qtype_col = df["question_type"].astype(str)
    df = df[(df["dataset"] == "arkitscenes") & (qtype_col.str.startswith((args.qtype)))]
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    print(f"Selected {len(df)} samples from CSV.")

    # load model
    model, tokenizer = load_model(args.model_path, device=args.device)

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

    mcq = args.qtype in ["object_rel_direction", "object_rel_distance", "obj_appearance_order", "route_planning"]
    nq  = args.qtype in ["object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"]

    n_correct = 0
    n_total = 0
    relerrs: List[float] = []

    with open(args.out, "a", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sid = row.get("id")
            if sid in seen_ids:
                continue

            scene_name = str(row["scene_name"]).strip()
            image_path = find_scene_topdown_path(args.topdown_dir, scene_name)

            if not image_path:
                # skip if no image found; still write a stub record for traceability
                record = {
                    "id": sid,
                    "dataset": row["dataset"],
                    "scene_name": scene_name,
                    "question_type": row["question_type"],
                    "question": str(row["question"]),
                    "skip_reason": "topdown_image_not_found",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # load image tiles
            pixel_values = load_topdown_image_tensor(
                image_path, input_size=args.input_size, max_num=args.max_tiles, device=args.device
            )

            q = str(row["question"])
            options_text = str(row.get("options", ""))
            gt = str(row.get("ground_truth", "")).strip()

            out = answer(
                model=model,
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=q,
                qtype="mcq" if mcq else "nq",
                cot=args.cot,
                options_text=options_text,
                gen_cfg=dict(max_new_tokens=1024, do_sample=False, temperature=0.0),
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
                    "image_path": image_path,
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
                    "image_path": image_path,
                }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ----------------------------
    # Summary
    # ----------------------------
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

    if mcq:
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

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


# python VSI-Bench/InternVL-8B/image_info.py --out VSI-Bench/InternVL-8B/results/3.5/object_rel_direction/image_info.json --qtype object_rel_direction \
# --model_path OpenGVLab/InternVL3_5-8B

# python VSI-Bench/InternVL-8B/image_info.py --out VSI-Bench/InternVL-8B/results/3.5/object_rel_distance/image_info.json --qtype object_rel_distance \
# --model_path OpenGVLab/InternVL3_5-8B

# python VSI-Bench/InternVL-8B/image_info.py --out VSI-Bench/InternVL-8B/results/3.5/route_planning/image_info.json --qtype route_planning \
# --model_path OpenGVLab/InternVL3_5-8B

# python VSI-Bench/InternVL-8B/image_info.py --out VSI-Bench/InternVL-8B/results/3.5/object_abs_distance/image_info.json --qtype object_abs_distance \
# --model_path OpenGVLab/InternVL3_5-8B