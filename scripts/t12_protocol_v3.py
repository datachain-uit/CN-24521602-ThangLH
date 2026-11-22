"""T12: áp giao thức đánh giá v3 lên kết quả T7 có sẵn (không chạy lại B0/B1/TRIV) và ghi reports/T12/*.csv cùng run.json."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t7 import git_info, package_versions  # noqa: E402  (cùng kiểu manifest với T7)
from src.eval.protocol_v3 import compare, f1_floor, loco_choice, normalized_gain, seed_average  # noqa: E402

KEYS = ["campaign", "q", "episode"]
METRICS = ["f1_star", "homogeneity", "wp", "nmi"]
NOTES = {
    "n_clusters_ge2": "T7 không lưu phân hoạch; partition_metrics bỏ phân hoạch sau khi tính chỉ số; sẽ có từ lần chạy P6/VC. Cột n_clusters_ge2 trong t12_summary.csv để NaN.",
    "ds2": "DS2 loại khỏi T12: không có dòng P5 (p5_baselines.csv) cho DS2 nên không có lựa chọn TRIV_LOCO.",
    "bootstrap": "Thứ tự đợt = episode tăng dần; RNG = np.random.default_rng(seed); một lần gọi integers(0, n, (n_boot, n)) cho mỗi (chiến dịch, q, cặp, trọng số). CI lệch 0,001–0,013 so với số GVHD được chấp nhận, không chỉnh code.",
    "eligible": "Chỉ dùng dòng eligible=True của T7 (≥ 5 IO và ≥ 5 control tại tau).",
    "seed_average": "Trung bình theo seed trước; floor và G tính theo từng đợt rồi mới trung bình trên đợt. *_seed_std là độ lệch chuẩn mẫu (ddof = 1) giữa seed, NaN với TRIV (1 dòng/mốc).",
    "triv": "TRIV_LOCO = dòng TRIV_{điểm}_{chiều} của T7 với tổ hợp LOCO của chiến dịch; TRIV_best = dòng TRIV_best của T7 (max trên 6 tổ hợp tại từng đợt-mốc), chỉ là cận trên tham chiếu.",
    "weights": "Trọng số độ nhạy = io_actions_episode (tổng hành động IO của đợt) nối từ p4_checkpoints.csv (checkpoint_type = io_pct), thống nhất với P4b của T9.",
    "role": "role = 'khẳng định' nếu q = q_main, ngược lại 'thăm dò'.",
}


def sha256(path):
    """sha256 của một tệp."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(paths, cfg):
    """Đọc T7, P5 và P4 theo tên tệp trong configs/t12.json và thư mục trong configs/paths.json."""
    files = {"t7_results": ROOT / paths["t7_output_dir"] / cfg["inputs"]["t7_results"], "p5_baselines": ROOT / paths["t9_output_dir"] / cfg["inputs"]["p5_baselines"], "p4_checkpoints": ROOT / paths["t9_output_dir"] / cfg["inputs"]["p4_checkpoints"]}
    return {name: pd.read_csv(path) for name, path in files.items()}, files


def loco_table(p5, campaigns, prefix):
    """Lựa chọn LOCO cho mọi chiến dịch của P5, kèm F1* trung bình của tổ hợp được chọn trên các chiến dịch khác."""
    rows = []
    for campaign in campaigns:
        score, direction = loco_choice(p5, campaign)
        other = p5[p5["campaign"].ne(campaign) & p5["eligible"].astype(bool)]
        chosen = other[other["method"].eq(score) & other["direction"].eq(direction)]
        rows.append(dict(campaign=campaign, score=score, direction=direction, t7_method=f"{prefix}{score}_{direction}", mean_F1_star_other=chosen["F1_star"].mean(), n_rows_other=len(other), n_campaigns_other=other["campaign"].nunique()))
    return pd.DataFrame(rows)


def consistency_check(t7, p5, loco, cfg):
    """So F1* của tổ hợp LOCO giữa dòng TRIV của T7 và p5_baselines.csv trên các mốc io_pct có ở cả hai; chỉ báo, không sửa."""
    p5 = p5[p5["checkpoint_type"].eq(cfg["consistency_checkpoint_type"])]
    rows = []
    for item in loco.loc[loco["campaign"].isin(cfg["campaigns"])].to_dict("records"):
        ours = t7[t7["campaign"].eq(item["campaign"]) & t7["method"].eq(item["t7_method"])][KEYS + ["f1_star"]]
        theirs = p5[p5["campaign"].eq(item["campaign"]) & p5["method"].eq(item["score"]) & p5["direction"].eq(item["direction"])][KEYS + ["F1_star"]]
        both = ours.merge(theirs, on=KEYS, how="inner", validate="one_to_one")
        gap = (both["f1_star"] - both["F1_star"]).abs()
        rows.append(dict(campaign=item["campaign"], method=item["t7_method"], n_common=len(both), max_abs_diff=float(gap.max()) if len(both) else None, n_over_tolerance=int((gap >= cfg["consistency_tolerance"]).sum()), over_tolerance=both.loc[gap >= cfg["consistency_tolerance"], KEYS].to_dict("records")))
    return rows


def build_methods(t7, loco, cfg):
    """Dòng T7 eligible của các phương pháp v3: B0_K*, B1_goc, B1_chuan, TRIV_LOCO (đổi tên từ tổ hợp LOCO của chiến dịch), TRIV_best."""
    m = cfg["methods"]
    base = t7[t7["campaign"].isin(cfg["campaigns"]) & t7["eligible"].astype(bool)]
    keep = base[base["method"].isin([*m["b0"], m["b1_goc"], m["b1_chuan"], m["triv_best"]])]
    picks = dict(zip(loco["campaign"], loco["t7_method"]))
    triv = base[base["method"].eq(base["campaign"].map(picks))].assign(method=m["triv_loco"])
    combos = base[base["method"].str.startswith(m["triv_prefix"]) & base["method"].ne(m["triv_best"])].groupby(KEYS)["f1_star"].max()
    best = base[base["method"].eq(m["triv_best"])].set_index(KEYS)["f1_star"]
    if not np.allclose(best.sort_index(), combos.reindex(best.sort_index().index), rtol=0, atol=1e-12):
        raise ValueError("TRIV_best của T7 khác max trên các tổ hợp TRIV.")
    return pd.concat([keep, triv], ignore_index=True)


def per_episode_table(rows, p4, cfg):
    """Trung bình seed, nối trọng số io_actions_episode (assert không mất/nhân bản dòng), thêm floor và G."""
    table = seed_average(rows, keys=[*KEYS, "method"], metrics=METRICS, first=["tau", "n_io", "n_ctrl"]).rename(columns={"f1_star": "f1", "f1_star_seed_std": "f1_seed_std"})
    marks = p4[p4["checkpoint_type"].eq(cfg["consistency_checkpoint_type"])][KEYS + [cfg["weight_column"]]]
    if marks.duplicated(KEYS).any():
        raise ValueError("Khóa nối (campaign, q, episode) của p4_checkpoints.csv không duy nhất.")
    n_before = len(table)
    table = table.merge(marks, on=KEYS, how="left", validate="many_to_one")
    if len(table) != n_before or table[cfg["weight_column"]].isna().any():
        raise ValueError("Phép nối trọng số làm mất hoặc nhân bản dòng.")
    table["floor"] = f1_floor(table["n_io"], table["n_ctrl"])
    table["G"] = normalized_gain(table["f1"], table["floor"])
    order = [*KEYS, "tau", "method", "n_seeds", "n_io", "n_ctrl", cfg["weight_column"], "floor", "f1", "f1_seed_std", "G", "nmi", "nmi_seed_std", "homogeneity", "homogeneity_seed_std", "wp", "wp_seed_std"]
    return table[order].sort_values([*KEYS, "method"]).reset_index(drop=True)


def summary_table(per_episode, cfg):
    """Tóm tắt theo (chiến dịch, q, phương pháp), trọng số đều theo đợt."""
    out = per_episode.groupby(["campaign", "q", "method"]).agg(n_episodes=("episode", "nunique"), floor=("floor", "mean"), f1=("f1", "mean"), G=("G", "mean"), nmi=("nmi", "mean"), homogeneity=("homogeneity", "mean"), wp=("wp", "mean"), f1_seed_std_max=("f1_seed_std", "max")).reset_index()
    out.insert(out.columns.get_loc("nmi") + 1, "n_clusters_ge2", np.nan)
    out["role"] = np.where(np.isclose(out["q"], cfg["q_main"]), "khẳng định", "thăm dò")
    return out


def compare_table(per_episode, cfg):
    """So sánh theo cặp cho mọi chiến dịch × q × cặp × trọng số (đều = chính; io_actions_episode = độ nhạy)."""
    rows = []
    for (campaign, q), part in per_episode.groupby(["campaign", "q"]):
        for method, ref in cfg["pairs"]:
            for weighting, column in (("equal", None), (cfg["weight_column"], cfg["weight_column"])):
                result = compare(part, method, ref, delta=cfg["delta"], n_boot=cfg["n_boot"], seed=cfg["seed"], weights=column, min_episodes=cfg["min_episodes"])
                rows.append(dict(campaign=campaign, q=q, method=method, ref=ref, weighting=weighting, primary=[method, ref] == cfg["pairs"][0] and weighting == "equal", role="khẳng định" if np.isclose(q, cfg["q_main"]) else "thăm dò", **result))
    return pd.DataFrame(rows)


def seed_std_table(per_episode, cfg):
    """Tỷ lệ cặp (đợt, mốc) có std F1* giữa seed = 0, riêng từng phương pháp và chiến dịch."""
    part = per_episode[per_episode["method"].isin(cfg["seed_std_methods"])]
    zero = part["f1_seed_std"].le(cfg["std_zero_tolerance"])
    return part.assign(std_zero=zero).groupby(["campaign", "method"]).agg(n_pairs=("std_zero", "size"), n_std_zero=("std_zero", "sum"), frac_std_zero=("std_zero", "mean"), f1_seed_std_max=("f1_seed_std", "max")).reset_index()


def print_main(summary, compare_rows, cfg):
    """In bảng q = q_main, trọng số đều: tóm tắt và các cặp so sánh."""
    pd.set_option("display.width", 200)
    print(f"\n=== q = {cfg['q_main']:.0%}, trọng số đều: tóm tắt ===")
    print(summary[np.isclose(summary["q"], cfg["q_main"])].drop(columns=["role"]).round(3).to_string(index=False))
    print(f"\n=== q = {cfg['q_main']:.0%}, trọng số đều: so sánh cặp ===")
    print(compare_rows[np.isclose(compare_rows["q"], cfg["q_main"]) & compare_rows["weighting"].eq("equal")].drop(columns=["weighting", "role"]).round(3).to_string(index=False))


def main():
    for console in (sys.stdout, sys.stderr):
        console.reconfigure(encoding="utf-8")
    paths = json.loads((ROOT / "configs" / "paths.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / paths["t12_config"]).read_text(encoding="utf-8"))
    out_dir = ROOT / paths["t12_output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data, files = load_inputs(paths, cfg)
    t7, p5, p4 = data["t7_results"], data["p5_baselines"], data["p4_checkpoints"]
    loco = loco_table(p5, sorted(p5["campaign"].unique()), cfg["methods"]["triv_prefix"])
    check = consistency_check(t7, p5, loco, cfg)
    per_episode = per_episode_table(build_methods(t7, loco, cfg), p4, cfg)
    summary = summary_table(per_episode, cfg)
    compare_rows = compare_table(per_episode, cfg)
    seed_std = seed_std_table(per_episode, cfg)
    outputs = {"t12_loco_choice.csv": loco, "t12_per_episode.csv": per_episode, "t12_summary.csv": summary, "t12_compare.csv": compare_rows, "t12_seed_std.csv": seed_std}
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False)
    payload = dict(task="T12", started_utc=started_utc, updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), **git_info(), packages=package_versions(), inputs={name: dict(path=str(path.relative_to(ROOT)).replace("\\", "/"), sha256=sha256(path)) for name, path in files.items()}, config=cfg, loco_choice=loco[["campaign", "score", "direction", "t7_method"]].to_dict("records"), loco_consistency=check, notes=NOTES, outputs=sorted(outputs))
    (out_dir / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("LOCO:", loco[["campaign", "score", "direction"]].to_string(index=False), sep="\n")
    for item in check:
        print(f"Nhất quán LOCO {item['campaign']}: {item['n_common']} mốc chung, max |ΔF1*| = {item['max_abs_diff']}, {item['n_over_tolerance']} mốc vượt ngưỡng {cfg['consistency_tolerance']}.")
    print_main(summary, compare_rows, cfg)
    print(f"\nĐã ghi {', '.join(sorted(outputs))} và run.json vào {out_dir.relative_to(ROOT)}; code_dirty = {payload['code_dirty']}.")


if __name__ == "__main__":
    main()
