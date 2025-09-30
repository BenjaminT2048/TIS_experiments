#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from mra_eval import (
    parse_number, accumulate_mra_stats, evaluate_numeric_pair
)

# ----------------------------
# Image / video tiling helpers
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size: int):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video_as_tiles(
    video_path: str,
    bound: Optional[Tuple[float, float]] = None,
    input_size: int = 448,
    max_num_tiles: int = 1,
    num_segments: int = 8,
    device: str = "cuda"
) -> Tuple[Optional[torch.Tensor], List[int]]:
    """
    Return:
      pixel_values: shape [sum_tiles, 3, H, W] on device (or None if max_num_tiles <= 0)
      num_patches_list: list of tiles-per-frame
    """
    if max_num_tiles <= 0:
        return None, []

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    pixel_values_list, num_patches_list = [], []
    for fi in frame_indices:
        img = Image.fromarray(vr[fi].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num_tiles)
        tile_tensors = [transform(tile) for tile in tiles]
        tile_tensors = torch.stack(tile_tensors)  # [tiles, 3, H, W]
        num_patches_list.append(tile_tensors.shape[0])
        pixel_values_list.append(tile_tensors)

    pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).to(device)
    return pixel_values, num_patches_list

# ----------------------------
# Prompt blocks
# ----------------------------
SYSTEM_INSTR_mcq = (
    "You are given N video frames as tiled images. Use spatial reasoning carefully.\n"
    "Rules:\n"
    "1) Read the question and options (A/B/C/D).\n"
    "2) Do NOT show any reasoning or explanation.\n"
    "3) On the LAST line, output exactly one letter in [A,B,C,D] with no extra text."
)

SYSTEM_INSTR_mcq_cot = (
    "You are given N video frames as tiled images. Use spatial reasoning carefully.\n"
    "Rules:\n"
    "1) Read the question and options (A/B/C/D).\n"
    "2) Assume a local coordinate system exactly as described in the question.\n"
    "3) You are encourage to output your thinking process.\n"
    "4) On the LAST line, output exactly one letter in [A,B,C,D] with no extra text."
)

SYSTEM_INSTR_nq = (
    "You are given N video frames as tiled images. Use spatial reasoning carefully.\n"
    "Rules:\n"
    "1) Read the question.\n"
    "2) Assume a local coordinate system exactly as described in the question.\n"
    "3) Do NOT show any reasoning or explanation.\n"
    "4) On the LAST line, output exactly one number with no extra text."
)

SYSTEM_INSTR_nq_cot = (
    "You are given N video frames as tiled images. Use spatial reasoning carefully.\n"
    "Rules:\n"
    "1) Read the question.\n"
    "2) Assume a local coordinate system exactly as described in the question.\n"
    "3) You are encourage to output your thinking process.\n"
    "4) On the LAST line, output exactly one number with no extra text."
)

def build_prompt_bimodal(
    question: str,
    qtype: str,
    cot: bool,
    num_frames: int,
    options_text: str,
    obj2d_text: Optional[str] = None,
    obj2d_unit: str = "meters"
) -> str:
    """
    双模态 prompt：前半段列出帧占位符；随后嵌入 JSON 信息块；最后附上规则与问题/选项。
    """
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(max(1, num_frames))])  # 至少 1 个占位

    if qtype == "mcq":
        SYSTEM_INSTR = SYSTEM_INSTR_mcq_cot if cot else SYSTEM_INSTR_mcq
    elif qtype == "nq":
        SYSTEM_INSTR = SYSTEM_INSTR_nq_cot if cot else SYSTEM_INSTR_nq
    else:
        SYSTEM_INSTR = SYSTEM_INSTR_mcq  # 兜底

    obj2d_block = ""
    if obj2d_text:
        obj2d_block = (
            f"The JSON contains view-invariant 2D positions (in {obj2d_unit}) of objects in this scene. "
            "For each label, the list contains all instances (length = count; each [x,y] is one instance).\n"
            "<OBJECT_2D_INFO_JSON>\n"
            f"{obj2d_text}\n"
            "</OBJECT_2D_INFO_JSON>\n"
            "Integrate the provided information with the video when reasoning."
        )

    prompt = (
        f"{video_prefix}"
        f"{SYSTEM_INSTR}\n"
        f"{obj2d_block}"
        f"Question: {question.strip()}\n"
    )
    if qtype == "mcq" and options_text:
        prompt += f"Options: {options_text.strip()}\n"

    if cot:
        prompt += "You are encourage to show your thinking process. Output exactly your answer in the last line.\n"
    else:
        prompt += "Do not give any reasoning process. Output exactly your answer in the last line.\n"
    return prompt

# ----------------------------
# Parsers
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
        trust_remote_code=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

# ----------------------------
# Object-2D JSON loader (robust)
# ----------------------------
def find_scene_obj2d_path(obj2d_dir: Optional[str], scene_name: str) -> Optional[str]:
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
    scale_to_m: float = 0.01,  # 若原始单位是厘米，0.01 转米；若本身就是米，传 1.0
) -> Optional[str]:
    import json
    from decimal import Decimal, getcontext, ROUND_HALF_UP
    getcontext().prec = 28

    def _round_xy(xy, nd):
        # 忽略 nd，强制两位小数
        try:
            x_cm, y_cm = Decimal(str(xy[0])), Decimal(str(xy[1]))
            # 先做单位换算：若原始是厘米 -> 米
            x_m = (x_cm * Decimal(str(0.01)))  # 通常 scale_to_m=0.01
            y_m = (y_cm * Decimal(str(0.01)))
            # 定点量化到两位小数（四舍五入）
            x_q = x_m.quantize(Decimal("0.000"), rounding=ROUND_HALF_UP)
            y_q = y_m.quantize(Decimal("0.000"), rounding=ROUND_HALF_UP)
            return [format(x_q, "f"), format(y_q, "f")]
        except Exception:
            return None

    def _build_text(data, nd, keep_top_labels: Optional[int] = None):
        counts, grouped = {}, {}
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

        obj = {"coord_system": "view-invariant 2D (x,y)", "coords_by_label": coords_by_label}
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

# ----------------------------
# Bimodal answering (video + info)
# ----------------------------
@torch.no_grad()
def answer_on_video(
    model,
    tokenizer,
    video_path: str,
    question: str,
    options_text: str,
    qtype: str,
    cot: bool,
    num_segments: int = 32,
    max_num_tiles: int = 1,
    input_size: int = 448,
    gen_cfg: Optional[Dict] = None,
    device: str = "cuda",
    obj2d_text: Optional[str] = None,
    obj2d_unit: str = "meters",
) -> Dict:
    if gen_cfg is None:
        gen_cfg = dict(max_new_tokens=128, do_sample=False, temperature=0.0)

    pixel_values, num_patches_list = load_video_as_tiles(
        video_path=video_path,
        input_size=input_size,
        max_num_tiles=max_num_tiles,
        num_segments=num_segments,
        device=device
    )

    # 如果只想 info-only，可把 max_num_tiles 设为 0；上面会返回 None 和空列表
    num_frames = len(num_patches_list) if num_patches_list else 1

    prompt = build_prompt_bimodal(
        question=question,
        qtype=qtype,
        cot=cot,
        num_frames=num_frames,
        options_text=options_text,
        obj2d_text=obj2d_text,
        obj2d_unit=obj2d_unit
    )

    response = model.chat(
        tokenizer,
        pixel_values,                 # None => info-only; Tensor => video(+info)
        prompt,
        gen_cfg,
        num_patches_list=num_patches_list if pixel_values is not None else None,
    )

    return {"raw_response": response, "prompt": prompt}

# ----------------------------
# Main runner
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Bimodal baseline: video frames + injected object-2D info")
    ap.add_argument("--csv", type=str, default="VSI-Bench/test-00000-of-00001.csv")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_segments", type=int, default=16)
    ap.add_argument("--max_num_tiles", type=int, default=1, help="tiles per frame; set 0 for info-only")
    ap.add_argument("--input_size", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    ap.add_argument("--video_dir", type=str, default="videos/arkitscenes")
    ap.add_argument("--cot", type=bool, default=False)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--qtype", type=str, required=True, choices=[
        "object_rel_direction", "object_rel_distance", "obj_appearance_order", "route_planning",
        "object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"
    ])
    ap.add_argument("--inject_info", type=bool, default=True, help="inject OBJECT_2D_INFO_JSON into prompt")
    ap.add_argument("--obj2d_dir", type=str, default="ARKitScenes/2D_annotation")
    ap.add_argument("--obj2d_max_chars", type=int, default=8000)
    ap.add_argument("--obj2d_unit", type=str, default="meters", help="annotation unit name shown in prompt")
    ap.add_argument("--obj2d_scale_to_m", type=float, default=0.01, help="scale centroid_xy to meters (e.g., 0.01 for cm)")
    args = ap.parse_args()

    # load data
    df = pd.read_csv(args.csv)
    df = df[(df["dataset"] == "arkitscenes") & (df["question_type"].astype(str).str.startswith(args.qtype))]
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
            video_path = os.path.join(args.video_dir, f"{scene_name}.mp4")
            if args.max_num_tiles > 0 and not os.path.exists(video_path):
                print(f"[warn] missing video: {video_path}")
                continue

            # locate & load per-scene object-2D JSON
            obj2d_path = None
            obj2d_text = None
            if args.inject_info and args.obj2d_dir:
                obj2d_path = find_scene_obj2d_path(args.obj2d_dir, scene_name)
                if obj2d_path:
                    obj2d_text = load_obj2d_as_prompt_text(
                        obj2d_path, max_chars=args.obj2d_max_chars, scale_to_m=args.obj2d_scale_to_m
                    )
                    if not obj2d_text:
                        try:
                            obj2d_text = open(obj2d_path, "r", encoding="utf-8", errors="ignore").read()[:args.obj2d_max_chars]
                        except Exception:
                            obj2d_text = None

            q = str(row["question"])
            options_text = str(row.get("options", ""))
            gt = str(row.get("ground_truth", "")).strip()

            out = answer_on_video(
                model, tokenizer,
                video_path=video_path,
                question=q,
                options_text=options_text,
                qtype="mcq" if mcq else "nq",
                cot=args.cot,
                num_segments=args.num_segments,
                max_num_tiles=args.max_num_tiles,
                input_size=args.input_size,
                gen_cfg=dict(max_new_tokens=128, do_sample=False, temperature=0.0),
                device=device,
                obj2d_text=obj2d_text,
                obj2d_unit=args.obj2d_unit
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
                    "video_path": video_path if args.max_num_tiles > 0 else None,
                    "raw_response": out["raw_response"],
                    "obj2d_path": obj2d_path,
                    "obj2d_in_prompt": bool(obj2d_text),
                }
            else:
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
                    "video_path": video_path if args.max_num_tiles > 0 else None,
                    "obj2d_path": obj2d_path,
                    "obj2d_in_prompt": bool(obj2d_text),
                }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- Summary ----
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
        summary_rec = {
            "type": "summary",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "file": args.out,
            "dataset": "arkitscenes",
            "qtypes": sorted(df["question_type"].astype(str).unique().tolist()),
            "metrics": {
                "nq": {
                    "n_valid": int(len(relerrs)),
                    "mean_mra": _clean_float(stats.mean_mra if stats is not None else None),
                    "per_threshold_acc": {
                        f"{t:.2f}": _clean_float(a) for t, a in (stats.per_threshold_acc.items() if stats is not None else {})
                    },
                    "mean_relative_error": _clean_float(float(np.mean(relerrs)) if len(relerrs) else None),
                },
            },
            "mra_thresholds": [float(t) for t in getattr(__import__('mra_eval'), 'MRA_THRESHOLDS', [])],
            "args": vars(args),
        }

    with open(args.out, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


# python VSI-Bench/InternVL-8B/integrated.py --out VSI-Bench/InternVL-8B/results/3.5/object_rel_direction/integrated.json --qtype object_rel_direction \
# --model_path OpenGVLab/InternVL3_5-8B


# python VSI-Bench/InternVL-8B/integrated.py --out VSI-Bench/InternVL-8B/results/3.5/object_rel_distance/integrated.json --qtype object_rel_distance \
# --model_path OpenGVLab/InternVL3_5-8B