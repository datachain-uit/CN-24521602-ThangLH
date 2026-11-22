"""T12 – kiểm tra phá hòa: (1) AUC của thứ tự phá hòa theo ID gốc mà P5 dùng so với nhãn IO; (2) độ nhạy của TRIV_LOCO với R hoán vị bí danh như T7. Không chạy lại B0/B1, không sửa P5."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t7 import git_info, load_ds1, package_versions  # noqa: E402
from src.eval.protocol_v3 import compare  # noqa: E402
from src.eval.seckin_protocol import baseline_scores, observed_at  # noqa: E402
from src.eval.t7_checkpoints import _rows_from_scores, load_config as load_t7_config, observe  # noqa: E402

KEYS = ["campaign", "q", "episode"]
Mark = namedtuple("Mark", ["episode", "tau"])
NOTES = {
    "p5_auc": "Thứ tự phá hòa của P5 = oracle_top_k: sort_index (accountid tăng dần) rồi sắp ổn định, nên trong mỗi nhóm hòa tài khoản ID nhỏ hơn đứng trước ở cả hai chiều. AUC = roc_auc_score(nhãn IO, −vị trí theo ID tăng dần) trên tập đánh giá tại mỗi mốc io_pct đạt; 0,5 = độc lập nhãn, > 0,5 = IO có ID nhỏ hơn.",
    "triv_tiebreak": "F1* của tổ hợp LOCO tính lại bằng _rows_from_scores của T7 (bí danh _alias_ids + oracle_top_k của T9) với seed hoán vị 0–19; điểm và tập đánh giá lấy như T7 (observe theo đợt tại tau). Seed triv_tiebreak_seed của T7 dùng để kiểm tái tạo t7_results.csv.",
    "compare": "TRIV_LOCO_mean20 = trung bình F1* qua các hoán vị tại từng đợt-mốc; B1_chuan lấy từ t12_per_episode.csv; bootstrap như t12_compare.csv. verdict_t7 là kết luận B1_chuan vs TRIV_LOCO trong t12_compare.csv.",
}


def sha256(path):
    """sha256 của một tệp."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def p5_tiebreak_auc(stream, marks):
    """AUC của thứ tự ID tăng dần (thứ tự phá hòa P5) với nhãn IO tại từng mốc io_pct đạt, tập đánh giá như evaluate_baselines của T9."""
    values = []
    for mark in marks.loc[marks["eligible"].astype(bool)].to_dict("records"):
        _, current = observed_at(stream, Mark(mark["episode"], mark["tau"]))
        labels = (~current.groupby("accountid", sort=True)["is_control"].first()).sort_index().astype(int)
        values.append(roc_auc_score(labels.to_numpy(), -np.arange(len(labels))))
    return np.asarray(values)


def triv_permutations(stream, marks, campaign, score, t7_method, seeds, t7_seed, t7_rows, tolerance):
    """F1* của tổ hợp LOCO với từng seed hoán vị; kiểm seed của T7 tái tạo đúng t7_results.csv."""
    rows = []
    for mark in marks.loc[marks["eligible"].astype(bool)].to_dict("records"):
        _, current = observe(stream, mark["tau"], mark["episode"])
        scores = baseline_scores(current)[[score]]
        labels = ~current.groupby("accountid")["is_control"].first()
        pick = lambda seed: next(row["f1_star"] for row in _rows_from_scores(scores, labels, seed) if row["method"] == t7_method)
        reproduced = pick(t7_seed)
        stored = t7_rows.loc[(campaign, mark["q"], mark["episode"])]
        if abs(reproduced - stored) >= tolerance:
            raise AssertionError(f"Hoán vị T7 không tái tạo F1* của {t7_method} tại {campaign} đợt {mark['episode']} q={mark['q']}: {reproduced} khác {stored}.")
        f1 = np.array([pick(seed) for seed in seeds])
        rows.append(dict(campaign=campaign, q=mark["q"], episode=mark["episode"], method=t7_method, n_permutations=len(f1), f1_t7=stored, f1_t7_reproduced=reproduced, f1_mean=f1.mean(), f1_min=f1.min(), f1_max=f1.max(), f1_range=f1.max() - f1.min(), f1_std=f1.std(ddof=1)))
    return rows


def compare_tiebreak(per_episode, triv, main_compare, cfg):
    """compare(B1_chuan vs TRIV_LOCO_mean20) mọi chiến dịch × q, trọng số đều và io_actions_episode; đối chiếu kết luận với t12_compare.csv."""
    tb, weight = cfg["tiebreak"], cfg["weight_column"]
    b1 = per_episode.loc[per_episode["method"].eq(tb["method"]), [*KEYS, "method", "f1", weight]]
    ref = triv[KEYS].assign(method=tb["ref"], f1=triv["f1_mean"]).merge(b1[[*KEYS, weight]], on=KEYS, how="left", validate="one_to_one")
    if ref[weight].isna().any() or len(ref) != len(b1):
        raise ValueError("Đợt-mốc của TRIV_LOCO_mean20 không khớp B1_chuan trong t12_per_episode.csv.")
    both = pd.concat([b1, ref], ignore_index=True)
    old = main_compare[main_compare["method"].eq(tb["method"]) & main_compare["ref"].eq(cfg["methods"]["triv_loco"])].set_index(["campaign", "q", "weighting"])
    rows = []
    for (campaign, q), part in both.groupby(["campaign", "q"]):
        for weighting, column in (("equal", None), (weight, weight)):
            result = compare(part, tb["method"], tb["ref"], delta=cfg["delta"], n_boot=cfg["n_boot"], seed=cfg["seed"], weights=column, min_episodes=cfg["min_episodes"])
            before = old.loc[(campaign, q, weighting)]
            rows.append(dict(campaign=campaign, q=q, method=tb["method"], ref=tb["ref"], weighting=weighting, role="khẳng định" if np.isclose(q, cfg["q_main"]) else "thăm dò", **result, mean_t7=before["mean"], lo_t7=before["lo"], hi_t7=before["hi"], verdict_t7=before["verdict"], verdict_changed=result["verdict"] != before["verdict"]))
    return pd.DataFrame(rows)


def main():
    for console in (sys.stdout, sys.stderr):
        console.reconfigure(encoding="utf-8")
    clock = perf_counter()
    paths = json.loads((ROOT / "configs" / "paths.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / paths["t12_config"]).read_text(encoding="utf-8"))
    t7_cfg = load_t7_config()
    tb, out_dir = cfg["tiebreak"], ROOT / paths["t12_output_dir"]
    files = {"t7_results": ROOT / paths["t7_output_dir"] / cfg["inputs"]["t7_results"], **{name: out_dir / file for name, file in tb["main_outputs"].items()}}
    t7 = pd.read_csv(files["t7_results"])
    per_episode, main_compare, loco = (pd.read_csv(files[name]) for name in ("per_episode", "compare", "loco_choice"))
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    auc_rows, triv_rows = [], []
    for item in loco.to_dict("records"):
        campaign = item["campaign"]
        stream, marks = load_ds1(campaign, t7_cfg, paths)
        auc = p5_tiebreak_auc(stream, marks)
        auc_rows.append(dict(campaign=campaign, n_marks=len(auc), auc_mean=auc.mean(), auc_min=auc.min(), auc_max=auc.max(), auc_std=auc.std(ddof=1)))
        if campaign in cfg["campaigns"]:
            t7_rows = t7.loc[t7["campaign"].eq(campaign) & t7["method"].eq(item["t7_method"]) & t7["eligible"].astype(bool)].set_index(KEYS)["f1_star"]
            rows = triv_permutations(stream, marks, campaign, item["score"], item["t7_method"], tb["permutation_seeds"], t7_cfg["triv_tiebreak_seed"], t7_rows, cfg["consistency_tolerance"])
            if len(rows) != len(t7_rows):
                raise ValueError(f"Số mốc đạt của {campaign} khác t7_results.csv.")
            triv_rows.extend(rows)
        print(f"[{campaign}] {len(auc)} mốc io_pct đạt, AUC phá hòa P5 trung bình {auc.mean():.3f} [{auc.min():.3f}; {auc.max():.3f}] | {perf_counter() - clock:.0f}s", flush=True)
    auc_table, triv = pd.DataFrame(auc_rows), pd.DataFrame(triv_rows).sort_values(KEYS).reset_index(drop=True)
    tie_compare = compare_tiebreak(per_episode, triv, main_compare, cfg)
    outputs = {tb["outputs"]["p5_auc"]: auc_table, tb["outputs"]["triv"]: triv, tb["outputs"]["compare"]: tie_compare}
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False)
    payload = dict(task="T12-tiebreak", started_utc=started_utc, updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), runtime_s=round(perf_counter() - clock, 1), **git_info(), packages=package_versions(), inputs={name: dict(path=str(path.relative_to(ROOT)).replace("\\", "/"), sha256=sha256(path)) for name, path in files.items()}, config=cfg, t7_triv_tiebreak_seed=t7_cfg["triv_tiebreak_seed"], t7_reproduction=dict(n_checked=len(triv), max_abs_diff=float((triv["f1_t7_reproduced"] - triv["f1_t7"]).abs().max()), tolerance=cfg["consistency_tolerance"]), notes=NOTES, outputs=sorted(outputs))
    (out_dir / tb["outputs"]["run"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 200)
    print("\n=== AUC phá hòa P5 ===", auc_table.round(3).to_string(index=False), sep="\n")
    print("\n=== Độ rộng F1* TRIV_LOCO qua hoán vị (theo q) ===", triv.groupby(["campaign", "q"]).agg(n=("episode", "size"), range_mean=("f1_range", "mean"), range_max=("f1_range", "max"), f1_t7=("f1_t7", "mean"), f1_mean20=("f1_mean", "mean")).round(3).to_string(), sep="\n")
    print("\n=== B1_chuan vs TRIV_LOCO_mean20 ===", tie_compare.drop(columns=["method", "ref", "role"]).round(3).to_string(index=False), sep="\n")
    print(f"\nTái tạo T7: {payload['t7_reproduction']}; code_dirty = {payload['code_dirty']}; {perf_counter() - clock:.0f}s.")


if __name__ == "__main__":
    main()
