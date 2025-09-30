# mra_eval.py
from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List

# 与主程序相同的数值正则，保证自包含
NUMBER_RE = re.compile(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?')

# θ ∈ {0.50, 0.55, ..., 0.95}
MRA_THRESHOLDS: List[float] = [0.50 + 0.05 * i for i in range(10)]
_EPS = 1e-12

def parse_number(text: Optional[str]) -> Optional[float]:
    """从任意文本解析首个数字（含小数/科学计数法）；失败返回 None。"""
    if not isinstance(text, str):
        return None
    m = NUMBER_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def relative_error(pred: float, gt: float) -> float:
    """|pred-gt| / max(|gt|, eps)；对 y=0 稳健。"""
    return abs(pred - gt) / max(abs(gt), _EPS)

def mra_from_relerr(relerr: float) -> float:
    """对单样本，在 10 个阈值上取指示函数的平均。"""
    hits = [(1.0 if relerr < (1.0 - th) else 0.0) for th in MRA_THRESHOLDS]
    return float(np.mean(hits))

@dataclass
class MRAStats:
    n_valid: int
    mean_mra: float
    per_threshold_acc: Dict[float, float]

def accumulate_mra_stats(relerrs: List[float]) -> MRAStats:
    """聚合一批相对误差，返回整体 MRA 与每阈值命中率。"""
    if len(relerrs) == 0:
        return MRAStats(0, float('nan'), {th: float('nan') for th in MRA_THRESHOLDS})
    arr = np.asarray(relerrs, dtype=np.float64)[:, None]       # (N,1)
    th  = np.asarray(MRA_THRESHOLDS, dtype=np.float64)[None,:] # (1,10)
    hits = (arr < (1.0 - th)).astype(np.float64)               # (N,10)
    per_threshold = {float(t): float(hits[:, i].mean()) for i, t in enumerate(MRA_THRESHOLDS)}
    mean_mra = float(hits.mean(axis=1).mean())
    return MRAStats(len(relerrs), mean_mra, per_threshold)

def evaluate_numeric_pair(pred_text: Optional[str], gt_text: Optional[str]):
    """
    便捷函数：给预测与GT文本，返回 (pred_num, gt_num, relerr, sample_mra)；
    任一无法解析则全返回 None。
    """
    pred_num = parse_number(pred_text)
    gt_num   = parse_number(gt_text)
    if pred_num is None or gt_num is None:
        return None, None, None, None
    relerr = relative_error(pred_num, gt_num)
    return pred_num, gt_num, relerr, mra_from_relerr(relerr)
