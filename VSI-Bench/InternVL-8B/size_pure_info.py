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
               
    obj2d_block = ""
    if obj2d_text:
        obj2d_block = (
            "The JSON contains per-object size triplets (axesLengths) [x,y,z]. "
            "For each label, the list contains all instances (length = count; each [x,y,z] is one instance). "
            "Values are measured in the centimeters.\n"
            "<OBJECT_SIZE_INFO_JSON>\n"
            f"{obj2d_text}\n"
            "</OBJECT_SIZE_INFO_JSON>\n"
        )


    prompt = (
        f"{SYSTEM_INSTR}\n"
        f"{obj2d_block}\n"
        f"Question: {question.strip()}\n"
    )
    if qtype == "mcq":
        f"Options: {options_text.strip()}\n"


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
    # fallback: last match anywhere
    ms = list(CHOICE_RE.finditer(text))
    return ms[-1].group(1).upper() if ms else None

# 允许：-3, +3, 3, 3., .5, 3.14, 1e-3, -2.0E+5 等
NUMBER_RE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')

def extract_numerical_answer(text: str) -> Optional[str]:
    """
    从模型输出中抓取最后一行中的数字答案（可能是整数/小数/科学计数法）。
    只看最后一行：若最后一行没有数字，返回 None。
    """
    if not isinstance(text, str):
        return None
    # 取最后一个非空行
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
    读取“尺寸版”JSON（推荐）：
      {
        "source_file": "...",            # 可选
        "axes_by_label": {               # 或 "sizes_by_label"
          "chair": [[x,y,z], [x,y,z], ...],
          "table": [[x,y,z], ...],
          ...
        }
      }

    也兼容旧版列表结构（含 label / centroid_xy），作为兜底。

    输出紧凑 JSON（仅包含需要的信息）：
      {"coord_system":"axesLengths [x,y,z]",
       "sizes_by_label":{"chair":[[x,y,z],...], ...}}
    """
    import os, json

    if not json_path or not os.path.isfile(json_path):
        return None

    # ---------- helpers ----------
    def _round_list(nums, nd):
        if nd is None:
            return [float(v) for v in nums]
        return [round(float(v), nd) for v in nums]

    def _from_sizes_dict(obj, nd) -> Optional[dict]:
        """
        从 {"axes_by_label":{...}} 或 {"sizes_by_label":{...}} 提取。
        """
        sizes = None
        if isinstance(obj, dict):
            if isinstance(obj.get("axes_by_label"), dict):
                sizes = obj["axes_by_label"]
            elif isinstance(obj.get("sizes_by_label"), dict):
                sizes = obj["sizes_by_label"]
        if not isinstance(sizes, dict):
            return None

        out = {}
        # 统一 label 为 str，值为 [[x,y,z], ...]，可四舍五入
        for lbl, arr in sizes.items():
            if not isinstance(arr, list):
                continue
            cleaned = []
            for triplet in arr:
                if (isinstance(triplet, (list, tuple)) and len(triplet) == 3):
                    cleaned.append(_round_list(triplet, nd))
            if cleaned:
                out[str(lbl)] = cleaned
        if not out:
            return None
        return {
            "coord_system": "axesLengths [x,y,z]",
            "sizes_by_label": out
        }

    def _from_old_list(obj, nd) -> Optional[dict]:
        """
        兼容旧版：data: [ {label, centroid_xy:[x,y], ...}, ... ]
        这里仅把二维坐标塞成“sizes_by_label”，用于兜底，不做单位缩放。
        """
        data = obj.get("data", []) if isinstance(obj, dict) else None
        if not isinstance(data, list):
            return None
        grouped = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            lbl = item.get("label")
            xy  = item.get("centroid_xy")
            if lbl is None or not (isinstance(xy, (list, tuple)) and len(xy) >= 2):
                continue
            # 用 [x,y] 填充（第三个值置 0），只是兜底
            grouped.setdefault(str(lbl), []).append(_round_list([xy[0], xy[1], 0.0], nd))
        if not grouped:
            return None
        return {
            "coord_system": "pseudo-axes [x,y,0]",
            "sizes_by_label": grouped
        }

    def _dump_fit(obj_dict: dict, limit: int) -> Optional[str]:
        """
        把 obj_dict 压到不超过 limit 字符；如果太长，按 label 数量裁剪。
        """
        if obj_dict is None:
            return None

        # 先试完整
        txt = json.dumps(obj_dict, ensure_ascii=False, separators=(",", ":"))
        if len(txt) <= limit:
            return txt

        # 裁剪 labels 个数
        sizes = obj_dict.get("sizes_by_label", {})
        if not isinstance(sizes, dict) or not sizes:
            # 没内容就尝试截断原串
            return (txt[:limit-20] + "...(truncated)") if len(txt) > limit else txt

        labels = sorted(sizes.keys())
        # 逐步减半裁剪
        for keep in [len(labels)//2, len(labels)//4, len(labels)//6, 10, 5, 3, 1]:
            keep = max(1, min(keep, len(labels)))
            new_sizes = {k: sizes[k] for k in labels[:keep]}
            candidate = {
                "coord_system": obj_dict.get("coord_system", "axesLengths [x,y,z]"),
                "sizes_by_label": new_sizes
            }
            t = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
            if len(t) <= limit:
                return t

        # 仍旧太长，最后硬截断
        return (t[:limit-20] + "...(truncated)") if len(t) > limit else t

    # ---------- load & build ----------
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        # 读不出来就返回截断的原文
        try:
            raw = open(json_path, "r", encoding="utf-8", errors="ignore").read().strip()
            return raw[:max_chars - 20] + "...(truncated)" if len(raw) > max_chars else raw
        except Exception:
            return None

    # 先按“尺寸版”解析；失败再走旧版兜底
    for nd in [decimals, 0, None]:
        parsed = _from_sizes_dict(obj, nd)
        if parsed is None:
            parsed = _from_old_list(obj, nd)
        if parsed is None:
            continue
        fitted = _dump_fit(parsed, max_chars)
        if fitted is not None:
            return fitted

    # 都不行时，返回最紧凑的原 JSON（截断）
    try:
        raw_min = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        return raw_min[:max_chars - 20] + "...(truncated)" if len(raw_min) > max_chars else raw_min
    except Exception:
        return None


# ----------------------------
# INFO-ONLY answering (no video)
# ----------------------------
@torch.no_grad()
def answer(
    model,
    tokenizer,
    question: str,
    qtype: str,
    cot: bool,
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


    prompt = build_prompt(
        question=question,
        obj2d_text=obj2d_text,
        qtype=qtype,
        cot=cot,
        options_text=options_text,
    )

    response = model.chat(
        tokenizer,
        None,
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
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("--csv", type=str, default="VSI-Bench/test-00000-of-00001.csv", help="Path to VSI-Bench CSV (converted from parquet).")
    ap.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--input_size", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    ap.add_argument("--obj2d_max_chars", type=int, default=8000, help="Max characters of object-2D JSON to inject into prompt (will be compacted/truncated safely).")
    ap.add_argument("--cot", type=bool, default=False)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--qtype", type=str, required=True, choices=["object_size_estimation", "room_size_estimation"])
    ap.add_argument("--obj2d_dir", type=str, default="ARKitScenes/LWH_annotation")

    args = ap.parse_args()

    # load data
    df = pd.read_csv(args.csv)
    qtype = df["question_type"].astype(str)
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
            # 兜底：文件存在但处理失败时，注入原文
            if obj2d_path and not obj2d_text:
                try:
                    obj2d_text = open(obj2d_path, "r", encoding="utf-8", errors="ignore").read()[:args.obj2d_max_chars]
                except Exception:
                    obj2d_text = None

            q = str(row["question"])
            options_text = str(row.get("options", ""))
            gt = str(row.get("ground_truth", "")).strip()


            out = answer(
                model, tokenizer,
                question=q,
                input_size=args.input_size,
                gen_cfg=dict(max_new_tokens=1024, do_sample=False, temperature=0.0),
                device=device,
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
    elif nq:
        stats = accumulate_mra_stats(relerrs)
        if stats.n_valid > 0:
            print(f"[NA] MRA on {stats.n_valid} numeric samples: {stats.mean_mra:.4f}")
            per_th = ", ".join([f"θ={t:.2f}:{a:.3f}" for t, a in stats.per_threshold_acc.items()])
            print(f"[NA] Per-threshold accuracy -> {per_th}")
        else:
            print("[NA] No numeric samples (or unparsable); MRA not computed.")
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


# python VSI-Bench/InternVL-8B/size.py --out VSI-Bench/InternVL-8B/results/3.5/object_size_estimation/pure_info.json --qtype object_size_estimation \
# --model_path OpenGVLab/InternVL3_5-8B


