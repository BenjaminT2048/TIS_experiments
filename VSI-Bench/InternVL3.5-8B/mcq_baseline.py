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
) -> Tuple[torch.Tensor, List[int]]:
    """
    Return:
      pixel_values: shape [sum_tiles, 3, H, W] on device
      num_patches_list: list of tiles-per-frame
    """
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
# Prompting for rel-direction
# ----------------------------
REL_DIR_SYSTEM_INSTR = (
    "You are given N video frames as tiled images. "
    "Use spatial reasoning carefully.\n"
    "Rules:\n"
    "1) Read the question and options (A/B/C/D).\n"
    "2) Do NOT show any reasoning or explanation.\n"
    "3) On the LAST line, output exactly one letter in [A,B,C,D] with no extra text."
)

# REL_DIR_SYSTEM_INSTR = (
#     "You are given N video frames as tiled images. "
#     "Use spatial reasoning carefully.\n"
#     "Rules:\n"
#     "1) Read the question and options (A/B/C/D).\n"
#     "2) Assume a local coordinate system exactly as described in the question.\n"
#     "3) You are encourage to output your thinking process.\n"
#     "4) On the LAST line, output exactly one letter in [A,B,C,D] with no extra text."
# )

def build_rel_dir_prompt(question: str, options_text: str, num_frames: int) -> str:
    """
    Build a prompt that enumerates frames and appends the Q&A instruction.
    The images themselves will be injected via '<image>' placeholders.
    """
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frames)])
    prompt = (
        f"{video_prefix}"
        f"{REL_DIR_SYSTEM_INSTR}\n\n"
        f"Question: {question.strip()}\n"
        f"Options: {options_text.strip()}\n"
        f"Output exactly one of A, B, C, or D in the last line."
    )
    return prompt
# f"You are encourage to output your thinking process. Output exactly one of A, B, C, or D in the last line."
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
    # fallback: last match anywhere
    ms = list(CHOICE_RE.finditer(text))
    return ms[-1].group(1).upper() if ms else None


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


@torch.no_grad()
def answer_rel_direction_on_video(
    model,
    tokenizer,
    video_path: str,
    question: str,
    options_text: str,
    num_segments: int = 32,
    max_num_tiles: int = 1,
    input_size: int = 448,
    gen_cfg: Optional[Dict] = None,
    device: str = "cuda"
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
    prompt = build_rel_dir_prompt(question, options_text, num_frames=len(num_patches_list))
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
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("--csv", type=str, default="VSI-Bench/test-00000-of-00001.csv", help="Path to VSI-Bench CSV (converted from parquet).")
    ap.add_argument("--video_dir", type=str, default="videos/arkitscenes", help="")
    ap.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_segments", type=int, default=16)
    ap.add_argument("--max_num_tiles", type=int, default=1, help="tiles per frame")
    ap.add_argument("--input_size", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--qtype", type=str, required=True)
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

    # prepare output
    seen_ids = set()
    if os.path.exists(args.out):
        # resume safely
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
            video_path = os.path.join(args.video_dir, f"{scene_name}.mp4")
            if not os.path.exists(video_path):
                print(f"[warn] missing video: {video_path}")
                continue

            q = str(row["question"])
            options_text = str(row.get("options", ""))
            gt = str(row.get("ground_truth", "")).strip()

            out = answer_rel_direction_on_video(
                model, tokenizer,
                video_path=video_path,
                question=q,
                options_text=options_text,
                num_segments=args.num_segments,
                max_num_tiles=args.max_num_tiles,
                input_size=args.input_size,
                gen_cfg=dict(max_new_tokens=128, do_sample=False, temperature=0.0),
                device=device
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
                "video_path": video_path,
                "raw_response": out["raw_response"]
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


# python VSI-Bench/InternVL3.5-8B/mcq_baseline.py --out VSI-Bench/InternVL3.5-8B/results/object_rel_direction/baseline2.json --qtype object_rel_direction
# python VSI-Bench/InternVL3.5-8B/mcq_baseline.py --out VSI-Bench/InternVL3.5-8B/results/object_rel_distance/baseline.json --qtype object_rel_distance