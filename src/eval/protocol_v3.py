"""Giao thức đánh giá v3 (T12): sàn F1, độ lợi chuẩn hóa G, lựa chọn TRIV theo LOCO, trung bình seed và so sánh cặp theo đợt bằng bootstrap.

Các hàm thuần, không đọc/ghi tệp; dùng lại cho P6 và VC.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DESCRIPTIVE = "chỉ mô tả"
WIN = "vượt"
LOSE = "thua"
INCONCLUSIVE = "không kết luận"


def f1_floor(n_io, n_ctrl):
    """F1 của tập "mọi tài khoản" (gán tất cả là IO): 2·N_IO / (2·N_IO + N_ctrl). Nhận số hoặc mảng/Series."""
    return 2 * n_io / (2 * n_io + n_ctrl)


def normalized_gain(f1, floor):
    """Độ lợi chuẩn hóa G = (F1 − sàn) / (1 − sàn): 0 khi F1 = sàn, 1 khi F1 = 1. Nhận số hoặc mảng/Series."""
    return (f1 - floor) / (1 - floor)


def loco_choice(p5: pd.DataFrame, campaign: str, score_col: str = "F1_star") -> tuple[str, str]:
    """Chọn (điểm, chiều) cho TRIV_LOCO của `campaign` chỉ từ các dòng eligible của các chiến dịch KHÁC.

    Với mỗi điểm (cột method) lấy chiều có F1* trung bình cao hơn, rồi lấy điểm có F1* trung bình cao nhất (mã tham chiếu GVHD).
    """
    other = p5[p5["campaign"].ne(campaign) & p5["eligible"].astype(bool)]
    if other.empty:
        raise ValueError(f"Không có dòng eligible nào ngoài {campaign}.")
    mean_f1 = other.groupby(["method", "direction"])[score_col].mean().unstack()
    direction = mean_f1.idxmax(axis=1)
    score = max(direction.index, key=lambda s: mean_f1.loc[s, direction[s]])
    return str(score), str(direction[score])


def seed_average(df: pd.DataFrame, keys: Sequence[str] = ("campaign", "q", "episode", "method"), metrics: Sequence[str] = ("f1_star", "homogeneity", "wp", "nmi"), first: Sequence[str] = ()) -> pd.DataFrame:
    """Trung bình các chỉ số theo seed trong từng nhóm `keys`; thêm `<metric>_seed_std` (độ lệch chuẩn mẫu giữa seed, chỉ để báo cáo; NaN nếu chỉ 1 seed) và `n_seeds`.

    Cột trong `first` (ví dụ n_io, n_ctrl, tau) phải hằng trong nhóm và được giữ nguyên.
    """
    groups = df.groupby(list(keys), sort=True, dropna=False)
    for column in first:
        if (groups[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"Cột {column} không hằng trong nhóm {list(keys)}.")
    agg = {**{column: (column, "first") for column in first}, "n_seeds": (list(metrics)[0], "size"), **{m: (m, "mean") for m in metrics}, **{f"{m}_seed_std": (m, "std") for m in metrics}}
    return groups.agg(**agg).reset_index()


def compare(per_episode: pd.DataFrame, method: str, ref: str, delta: float = 0.03, n_boot: int = 2000, seed: int = 42, weights: str | None = None, min_episodes: int = 5) -> dict:
    """So sánh cặp `method` − `ref` theo đợt (cột episode, method, f1) bằng bootstrap lấy lại đợt.

    Đợt được sắp tăng dần theo episode trước khi bootstrap; RNG = np.random.default_rng(seed), một lần gọi integers(0, n, (n_boot, n)).
    `weights`: tên cột trọng số theo đợt (ví dụ io_actions_episode); None = trọng số đều (tương đương mã GVHD). Có trọng số thì mean và mỗi mẫu bootstrap là trung bình có trọng số với cùng chỉ số đã lấy mẫu.
    Kết luận: < min_episodes (mặc định 5) đợt → "chỉ mô tả"; mean ≥ delta và lo > 0 → "vượt"; mean ≤ −delta và hi < 0 → "thua"; còn lại "không kết luận".
    Trả về dict n, mean, lo, hi, verdict.
    """
    wide = per_episode.pivot(index="episode", columns="method", values="f1").sort_index()
    diff = (wide[method] - wide[ref]).dropna()
    d = diff.to_numpy(dtype=float)
    w = None if weights is None else per_episode.groupby("episode")[weights].first().reindex(diff.index).to_numpy(dtype=float)
    if w is not None and (np.isnan(w).any() or (w < 0).any() or w.sum() <= 0):
        raise ValueError("Trọng số phải không âm, không NaN và có tổng dương.")
    n = len(d)
    mean = (d.mean() if w is None else np.average(d, weights=w)) if n else np.nan
    if n < min_episodes:
        return dict(n=n, mean=float(mean), lo=np.nan, hi=np.nan, verdict=DESCRIPTIVE)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    boot = d[idx].mean(axis=1) if w is None else (d[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    verdict = WIN if mean >= delta and lo > 0 else LOSE if mean <= -delta and hi < 0 else INCONCLUSIVE
    return dict(n=n, mean=float(mean), lo=float(lo), hi=float(hi), verdict=verdict)
