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
    obj3d_text: Optional[str] = None,
) -> str:
    """
    Build a prompt that contains exactly ONE <image> placeholder (blank) and the JSON block.
    The image is explicitly declared as blank; the model must rely ONLY on JSON.
    """

    if qtype == "mcq":
        SYSTEM_INSTR = SYSTEM_INSTR_mcq_cot if cot else SYSTEM_INSTR_mcq
    elif qtype == "nq":
        SYSTEM_INSTR = SYSTEM_INSTR_nq_cot if cot else SYSTEM_INSTR_nq
               
    

    obj3d_block = (
        "The JSON contains view-invariant 3D positions (in meters) of objects in this scene. "
        "For each label, the list contains all instances (length = count; each [x,y,z] is one instance).\n"
        "<OBJECT_3D_INFO_JSON>\n"
        f"{obj3d_text}\n"
        "</OBJECT_3D_INFO_JSON>\n"
    )


    prompt = (
        f"{SYSTEM_INSTR}\n"
        f"{obj3d_block}\n"
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
# Object-3d JSON loader (robust)
# ----------------------------
def find_scene_obj3d_path(obj3d_dir: Optional[str], scene_name: str) -> Optional[str]:
    """
    Find a JSON file in obj3d_dir whose filename starts with scene_name.
    Prefer exact '{scene_name}.json'; otherwise, the first prefix match.
    """
    if not obj3d_dir or not os.path.isdir(obj3d_dir):
        return None

    exact = os.path.join(obj3d_dir, f"{scene_name}.json")
    if os.path.isfile(exact):
        return exact

    for fname in os.listdir(obj3d_dir):
        if fname.startswith(scene_name) and fname.endswith(".json"):
            return os.path.join(obj3d_dir, fname)
    return None


def load_obj3d_as_prompt_text(
    json_path: str,
    max_chars: int = 8000,
    decimals: int = 2,
):
    """
    读取包含 3D 质心的 JSON，抽取为紧凑 JSON 字符串：
      {"coord_system":"view-invariant 3D (x,y,z)","coords_by_label":{"bed":[[x,y,z],...], ...}}
    规则与 3d 版一致：
      - 若某 label 在文件中出现次数 > 2，则整类删除；
      - 其余 (<=2) 全保留，并按 `decimals` 四舍五入。
    兼容结构（取到第一个可用者即停）：
      1) obj["data"][i]["segments"]["obb"]["centroid"] == [x,y,z]
      2) obj["data"][i]["segments"]["obbAligned"]["centroid"] == [x,y,z]
      3) obj["data"][i]["centroid"] / ["centroid_xyz"] == [x,y,z]
      4) 顶层是 list，每个 item 直接带 "centroid"/"centroid_xyz"
    """
    import json
    from collections import defaultdict

    def _round_xyz(xyz, nd):
        try:
            x, y, z = float(xyz[0])*0.01, float(xyz[1])*0.01, float(xyz[2])*0.01
        except Exception:
            return None
        # 处理 -0.0 -> 0.0
        def _r(v):
            rv = round(v, nd)
            if rv == -0.0:
                rv = 0.0
            return rv
        return [_r(x), _r(y), _r(z)]

    # 读文件
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # 归一化成一个可迭代的 items 列表
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        items = obj["data"]
    elif isinstance(obj, list):
        items = obj
    else:
        items = [obj]  # 兜底：单对象

    coords_by_label = defaultdict(list)

    def _extract_centroid3d(item):
        # 1) segments.obb.centroid
        try:
            return item["segments"]["obb"]["centroid"]
        except Exception:
            pass
        # 2) segments.obbAligned.centroid
        try:
            return item["segments"]["obbAligned"]["centroid"]
        except Exception:
            pass
        # 3) 扁平字段
        for k in ("centroid_xyz", "centroid"):
            if isinstance(item, dict) and k in item:
                return item[k]
        # 4) 有些结构把有用字段包在 "segments" 同级字段里
        if "segments" in item and isinstance(item["segments"], dict):
            for k in ("centroid_xyz", "centroid"):
                if k in item["segments"]:
                    return item["segments"][k]
        return None

    def _extract_label(item):
        # 常见位置：item["label"]
        if isinstance(item, dict):
            if "label" in item and isinstance(item["label"], str):
                return item["label"]
            # 有些数据会在更深层
            if "segments" in item and isinstance(item["segments"], dict):
                if "label" in item["segments"] and isinstance(item["segments"]["label"], str):
                    return item["segments"]["label"]
        return None

    # 遍历抽取
    for it in items:
        label = _extract_label(it)
        if not label:
            # 有些结构把 label 放在父级：例如 {"data":[{"label":...,"segments":{...}}]}
            # 这时 it 就是带 label 的；若拿不到就跳过
            continue
        c = _extract_centroid3d(it)
        if not (isinstance(c, (list, tuple)) and len(c) == 3):
            continue
        c_ = _round_xyz(c, decimals)
        if c_ is None:
            continue
        coords_by_label[label].append(c_)

    # 过滤：次数 > 2 的整类删除（与 3d 版保持一致）
    filtered = {
        lab: pts for lab, pts in coords_by_label.items()
        if len(pts) <= 2
    }

    compact = {
        "coord_system": "view-invariant 3D (x,y,z)",
        "coords_by_label": filtered
    }
    text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

    # 超长截断（与 3d 版同策略：简单截断）
    if len(text) > max_chars:
        text = text[:max_chars]

    return text



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
    obj3d_text: Optional[str] = None
) -> Dict:
    """
    不读取视频；提供一个全零的占位图像 (1x3xH x W)，并强制在 prompt 中声明“只用 JSON 推理”。
    """
    if gen_cfg is None:
        gen_cfg = dict(max_new_tokens=1024, do_sample=False, temperature=0.0)


    prompt = build_prompt(
        question=question,
        obj3d_text=obj3d_text,
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
    ap.add_argument("--obj3d_max_chars", type=int, default=8000, help="Max characters of object-3d JSON to inject into prompt (will be compacted/truncated safely).")
    ap.add_argument("--cot", type=bool, default=False)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--qtype", type=str, required=True, choices=["object_rel_direction", "object_rel_distance", "obj_appearance_order", "route_planning",
                                                                 "object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"])
    ap.add_argument("--obj3d_dir", type=str, default="/workspace/TIS_experiments/info/3D_annotation")

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
        mcq = args.qtype in ["object_rel_direction", "object_rel_distance", "obj_appearance_order", "route_planning"]
        nq = args.qtype in ["object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"]
        for _, row in tqdm(df.iterrows(), total=len(df)):
            
            sid = row.get("id")
            if sid in seen_ids:
                continue

            scene_name = str(row["scene_name"]).strip()

            # locate & load per-scene object-3d JSON
            obj3d_path = find_scene_obj3d_path(args.obj3d_dir, scene_name) if args.obj3d_dir else None
            obj3d_text = load_obj3d_as_prompt_text(obj3d_path, max_chars=args.obj3d_max_chars) if obj3d_path else None
            # 兜底：文件存在但处理失败时，注入原文
            if obj3d_path and not obj3d_text:
                try:
                    obj3d_text = open(obj3d_path, "r", encoding="utf-8", errors="ignore").read()[:args.obj3d_max_chars]
                except Exception:
                    obj3d_text = None

            q = str(row["question"])
            options_text = str(row.get("options", ""))
            gt = str(row.get("ground_truth", "")).strip()


            out = answer(
                model, tokenizer,
                question=q,
                input_size=args.input_size,
                gen_cfg=dict(max_new_tokens=1024, do_sample=False, temperature=0.0),
                device=device,
                obj3d_text=obj3d_text,
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
                    "obj3d_path": obj3d_path,
                    "obj3d_in_prompt": bool(obj3d_text),
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
                    "obj3d_path": obj3d_path,
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


# python VSI-Bench/InternVL-8B/pure_info.py --out VSI-Bench/InternVL-8B/results/3.5/route_planning/pure_info.json --qtype route_planning \
# --model_path OpenGVLab/InternVL3_5-8B 


# python VSI-Bench/InternVL-8B/pure_info.py --out VSI-Bench/InternVL-8B/results/3.5/object_rel_distance/pure_infokfjf.json --qtype object_rel_distance \
# --model_path OpenGVLab/InternVL3_5-8B


# python VSI-Bench/InternVL-8B/pure_info_3D.py --out VSI-Bench/InternVL-8B/results/3.5/object_abs_distance/pure_info_3D.json --qtype object_abs_distance \
# --model_path OpenGVLab/InternVL3_5-8B

