"""P6 + VC: đọc shard kết quả của run_p6_vc.py và tạo bảng báo cáo theo quy tắc chốt trước (configs/p6.json, configs/vc.json).

Ra: reports/P6/p6_dependence.csv, reports/P6/p6_control_clusters.csv, reports/VC/p5_vc_all_campaigns.csv, reports/VC/vc_compare_q25.csv, reports/VC/vc_loco_choice.csv.
Hàm T12 dùng lại: seed_average, compare, loco_choice, f1_floor, normalized_gain.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_p6_vc import P2_FILE, config_hash, git_state, load_all  # noqa: E402
from src.eval.p6_variants import COLUMNS, QUAL_COLUMNS  # noqa: E402
from src.eval.protocol_v3 import compare, f1_floor, loco_choice, normalized_gain, seed_average  # noqa: E402

METRICS = ["f1_star", "homogeneity", "wp", "nmi", "n_clusters_ge2"]
FIRST = ["tau", "n_io", "n_ctrl", "io_actions_episode"]
WEIGHTINGS = (("equal", None), ("io_actions_episode", "io_actions_episode"))
MARK = ["campaign", "episode", "checkpoint_type", "checkpoint_value"]


def read_results(directory, pattern):
    """Ghép mọi shard; chỉ số đọc dạng số, eligible dạng bool."""
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"Không có {pattern} trong {directory}.")
    frame = pd.concat([pd.read_csv(path, dtype={"protocol": str, "campaign": str, "variant": str, "method": str, "config_hash": str}) for path in files], ignore_index=True)
    for column in ("eligible", "eligible_goc", "eligible_vc", "repeated_sync"):
        if column in frame:
            frame[column] = frame[column].map({True: True, False: False, "True": True, "False": False})
    return frame


def check_run(directory, digest, allow_incomplete):
    """run.json: config_hash khớp, code_dirty = False và mọi chiến dịch complete (trừ --allow-incomplete cho chạy thử)."""
    run = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
    problems = [] if run.get("config_hash") == digest else ["config_hash khác config hiện tại"]
    if not allow_incomplete:
        problems += ["code_dirty = True"] if run.get("code_dirty") else []
        problems += [f"{campaign} chưa đủ khóa ({info})" for campaign, info in run.get("campaigns", {}).items() if not info.get("complete")]
    if problems:
        raise SystemExit(f"{directory}/run.json: " + "; ".join(problems))
    return run


def per_episode(rows, keys):
    """Trung bình seed theo (campaign, q, episode, *keys) trên dòng eligible; thêm floor và G theo từng đợt (như T12)."""
    table = seed_average(rows, keys=["campaign", "q", "episode", *keys], metrics=METRICS, first=FIRST)
    table["floor"] = f1_floor(table["n_io"], table["n_ctrl"])
    table["G"] = normalized_gain(table["f1_star"], table["floor"])
    return table.rename(columns={"f1_star": "f1"})


def means(part, suffix):
    """Trung bình đều theo đợt các chỉ số báo cáo (floor, F1, G, NMI, số cụm ≥ 2; H, WP phụ)."""
    return {f"{name}_{suffix}": float(part[name].mean()) if len(part) else np.nan for name in ("floor", "f1", "G", "nmi", "n_clusters_ge2", "homogeneity", "wp")}


def run_compare(table, method, ref, cfg, weights):
    return compare(table, method, ref, delta=cfg["delta"], n_boot=cfg["n_boot"], seed=cfg["seed"], weights=weights, min_episodes=cfg["min_episodes"])


# --- P6 ---
def p6_dependence(results, p6):
    """d_e = F1*(biến thể) − F1*(main) theo đợt; compare(delta = strong_drop): 'thua' = giảm mạnh. Thêm dòng ALL: phụ thuộc mạnh ⇔ ≥ 2/3 chiến dịch đủ điều kiện 'thua'."""
    rule = p6["interpretation"]
    cfg = dict(delta=rule["strong_drop"], n_boot=rule["n_boot"], seed=rule["bootstrap_seed"], min_episodes=rule["min_episodes"])
    rows_in = results.loc[results["protocol"].eq("P6") & results["eligible"].eq(True)]
    table = per_episode(rows_in, ["variant"]).rename(columns={"variant": "method"})
    main_q = table.loc[table["method"].eq("main") & np.isclose(table["q"], rule["q_main"])]
    counts = main_q.groupby("campaign")["episode"].nunique()
    eligible = {campaign: int(counts.get(campaign, 0)) >= rule["min_episodes"] for campaign in p6["campaigns"]}
    rows = []
    for (campaign, q), part in table.groupby(["campaign", "q"]):
        for variant in sorted(set(part["method"]) - {"main"}):
            both = part.loc[part["method"].isin(["main", variant])]
            both = both.loc[both.groupby("episode")["method"].transform("nunique").eq(2)]
            for weighting, column in WEIGHTINGS:
                result = run_compare(both, variant, "main", cfg, column)
                rows.append(dict(campaign=campaign, q=q, variant=variant, factor=rule["factors"].get(variant, ""), weighting=weighting, primary=weighting == "equal", role="khẳng định" if np.isclose(q, rule["q_main"]) else "thăm dò", campaign_eligible=eligible.get(campaign, False), n_main_episodes_q_main=int(counts.get(campaign, 0)), n_episodes=result["n"], mean_d=result["mean"], ci95_low=result["lo"], ci95_high=result["hi"], verdict=result["verdict"], strong_drop=result["verdict"] == "thua", **means(both.loc[both["method"].eq("main")], "main"), **means(both.loc[both["method"].eq(variant)], "variant")))
    out = pd.DataFrame(rows)
    overall = []
    for (q, variant, weighting), part in out.groupby(["q", "variant", "weighting"]):
        judged = part.loc[part["campaign_eligible"]]
        n, strong = len(judged), int(judged["strong_drop"].sum())
        overall.append(dict(campaign="ALL", q=q, variant=variant, factor=rule["factors"].get(variant, ""), weighting=weighting, primary=weighting == "equal", role="khẳng định" if np.isclose(q, rule["q_main"]) else "thăm dò", n_eligible_campaigns=n, n_strong_drop=strong, strong_dependence=bool(n and strong / n >= rule["strong_min_share"] - 1e-12), eligible_campaigns=",".join(judged["campaign"])))
    return pd.concat([out, pd.DataFrame(overall)], ignore_index=True)


def p6_control_clusters(directory, p6, digest):
    """Gộp bảng cụm control (seed, q của p6.qualitative) theo mọi đợt; xếp (−control, −tỷ lệ control, −kích thước, episode, hạng) và giữ top mỗi chiến dịch."""
    qual = p6["qualitative"]
    if not list(Path(directory).glob("control_clusters_*.csv")):
        return pd.DataFrame(columns=["campaign", "pooled_rank", *[c for c in QUAL_COLUMNS if c not in ("campaign", "config_hash", "tau")], "n_episodes_pooled"]).rename(columns={"rank": "rank_in_episode"})
    frame = read_results(directory, "control_clusters_*.csv")
    frame = frame.loc[frame["config_hash"].eq(digest) & frame["seed"].eq(qual["seed"]) & np.isclose(frame["q"], qual["q"])]
    frame = frame.sort_values(["campaign", "control_accounts", "control_share", "accounts", "episode", "rank"], ascending=[True, False, False, False, True, True], kind="stable")
    top = frame.groupby("campaign", sort=True).head(qual["top"]).copy()
    top.insert(1, "pooled_rank", top.groupby("campaign").cumcount() + 1)
    top = top.rename(columns={"rank": "rank_in_episode"}).drop(columns=["config_hash", "tau"])
    episodes = frame.groupby("campaign")["episode"].nunique().rename("n_episodes_pooled")
    return top.merge(episodes, on="campaign", how="left")


# --- VC ---
def p5_table(results, vc):
    """P5 (AUC, F1* tầm thường) theo chiến dịch × loại mốc × giá trị mốc × điểm/chiều: mỗi giao thức trên eligible của nó, và trên mốc eligible ở cả hai; kèm số mốc đổi eligibility."""
    p5 = results.loc[results["method"].str.startswith("P5_") & results["protocol"].isin(["GOC", "VC"])]
    vc_rows = p5.loc[p5["protocol"].eq("VC")]
    flags = vc_rows.drop_duplicates(MARK)[[*MARK, "eligible_goc", "eligible_vc"]]
    both_marks = flags.loc[flags["eligible_goc"].eq(True) & flags["eligible_vc"].eq(True), MARK]
    rows = []
    for (campaign, kind, value, method), part in p5.groupby(["campaign", "checkpoint_type", "checkpoint_value", "method"]):
        mark = flags.loc[flags["campaign"].eq(campaign) & flags["checkpoint_type"].eq(kind) & flags["checkpoint_value"].eq(value)]
        row = dict(campaign=campaign, checkpoint_type=kind, checkpoint_value=value, q=part["q"].iloc[0], method=method, score=method[3:].rsplit("_", 1)[0], direction=method.rsplit("_", 1)[1], main_set=campaign in vc["method_campaigns"], n_marks=len(mark), n_eligible_goc=int(mark["eligible_goc"].sum()), n_eligible_vc=int(mark["eligible_vc"].sum()), n_eligibility_changed=int((mark["eligible_goc"] != mark["eligible_vc"]).sum()), n_eligible_both=int((mark["eligible_goc"] & mark["eligible_vc"]).sum()))
        for protocol in ("GOC", "VC"):
            own = part.loc[part["protocol"].eq(protocol) & part["eligible"].eq(True)]
            shared = own.merge(both_marks, on=MARK)
            row.update({f"auc_{protocol.lower()}": own["auc"].mean(), f"f1_{protocol.lower()}": own["f1_star"].mean(), f"auc_{protocol.lower()}_both": shared["auc"].mean(), f"f1_{protocol.lower()}_both": shared["f1_star"].mean()})
        row["delta_auc_both"] = row["auc_vc_both"] - row["auc_goc_both"]
        rows.append(row)
    return pd.DataFrame(rows)


def p5_frame(results, protocol):
    """Dòng P5 của một giao thức ở dạng p5_baselines.csv của T9 (method = điểm, direction, F1_star, eligible) để gọi loco_choice của T12."""
    p5 = results.loc[results["protocol"].eq(protocol) & results["method"].str.startswith("P5_")]
    return p5.assign(method=p5["method"].str[3:].str.rsplit("_", n=1).str[0], direction=p5["method"].str.rsplit("_", n=1).str[1], F1_star=p5["f1_star"], eligible=p5["eligible"].eq(True))


def loco_rows(results, protocol, campaigns):
    """loco_choice (T12) trên P5 cùng giao thức của các chiến dịch khác (mọi mốc io_pct và theo ngày eligible, như T12)."""
    p5 = p5_frame(results, protocol)
    choices = {}
    for campaign in campaigns:
        try:
            choices[campaign] = loco_choice(p5, campaign)
        except ValueError:
            choices[campaign] = (None, None)
    return choices


def loco_table(results, vc, t12_loco):
    """Lựa chọn LOCO của mọi chiến dịch trong p5_campaigns ở GOC và VC, F1* trung bình của tổ hợp trên chiến dịch khác, và đối chiếu với t12_loco_choice.csv (T12, tag t12-v1)."""
    reference = t12_loco.set_index("campaign")
    rows = []
    for protocol in ("GOC", "VC"):
        p5 = p5_frame(results, protocol)
        for campaign, (score, direction) in loco_rows(results, protocol, vc["p5_campaigns"]).items():
            other = p5.loc[p5["campaign"].ne(campaign) & p5["eligible"]]
            chosen = other.loc[other["method"].eq(score) & other["direction"].eq(direction)]
            t12 = reference.loc[campaign] if campaign in reference.index else None
            rows.append(dict(protocol=protocol, campaign=campaign, score=score, direction=direction, triv_method=f"TRIV_{score}_{direction}" if score else None, mean_F1_star_other=chosen["F1_star"].mean(), n_rows_other=len(other), n_campaigns_other=other["campaign"].nunique(), t12_score=None if t12 is None else t12["score"], t12_direction=None if t12 is None else t12["direction"], t12_mean_F1_star_other=None if t12 is None else t12["mean_F1_star_other"], t12_n_rows_other=None if t12 is None else t12["n_rows_other"], same_choice_as_t12=None if t12 is None else bool(score == t12["score"] and direction == t12["direction"])))
    return pd.DataFrame(rows)


def method_table(results, protocol, loco, campaigns):
    """Bảng theo đợt (trung bình seed) cho B1_chuan, B1_goc, TRIV_best và TRIV_LOCO (dòng TRIV_{điểm}_{chiều} của chiến dịch đổi tên)."""
    base = results.loc[results["protocol"].eq(protocol) & results["campaign"].isin(campaigns) & results["eligible"].eq(True)]
    picks = {campaign: f"TRIV_{score}_{direction}" for campaign, (score, direction) in loco.items() if score is not None}
    triv = base.loc[base["method"].eq(base["campaign"].map(picks))].assign(method="TRIV_LOCO")
    keep = base.loc[base["method"].isin(["B1_chuan", "B1_goc", "TRIV_best"])]
    return per_episode(pd.concat([keep, triv], ignore_index=True), ["method"])


def vc_compare(results, all_results, p6, vc):
    """q_main: (1) VC: B1_chuan vs TRIV_LOCO (eligible_vc); (2) GOC: B1_chuan (P6 main) vs TRIV_LOCO; (3) trước/sau: B1_chuan VC − GOC trên (đợt, q) eligible ở cả hai."""
    cmp_cfg = vc["comparison"]
    cfg = dict(delta=cmp_cfg["delta"], n_boot=cmp_cfg["n_boot"], seed=cmp_cfg["bootstrap_seed"], min_episodes=cmp_cfg["min_episodes"])
    campaigns = vc["method_campaigns"]
    q_main = cmp_cfg["q_main"]
    loco = {protocol: loco_rows(all_results, protocol, vc["p5_campaigns"]) for protocol in ("VC", "GOC")}
    p6_main = results.loc[results["protocol"].eq("P6") & results["variant"].eq("main")].assign(protocol="GOC")
    tables = {"VC": method_table(results, "VC", loco["VC"], campaigns), "GOC": method_table(pd.concat([results.loc[results["protocol"].eq("GOC")], p6_main], ignore_index=True), "GOC", loco["GOC"], campaigns)}
    rows = []
    for protocol, table in tables.items():
        for campaign, part in table.loc[np.isclose(table["q"], q_main)].groupby("campaign"):
            score, direction = loco[protocol][campaign]
            for weighting, column in WEIGHTINGS:
                result = run_compare(part, cmp_cfg["method"], cmp_cfg["ref"], cfg, column) if score is not None else dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, verdict="thiếu LOCO (không có chiến dịch khác)")
                extra = {f"{key[:-2]}_{name}": value for name in ("B1_chuan", "B1_goc", "TRIV_LOCO", "TRIV_best") for key, value in means(part.loc[part["method"].eq(name)], "x").items()}
                rows.append(dict(comparison=f"{protocol}: {cmp_cfg['method']} vs {cmp_cfg['ref']}", protocol=protocol, campaign=campaign, q=q_main, weighting=weighting, primary=weighting == "equal", loco_score=score, loco_direction=direction, n=result["n"], mean_d=result["mean"], ci95_low=result["lo"], ci95_high=result["hi"], verdict=result["verdict"], **extra))
    flags = results.loc[results["protocol"].eq("VC")].drop_duplicates(["campaign", "episode", "q"])
    both = flags.loc[flags["eligible_goc"].eq(True) & flags["eligible_vc"].eq(True), ["campaign", "episode", "q"]]
    pair = pd.concat([tables["VC"].assign(method=lambda d: d["method"] + "@VC"), tables["GOC"].assign(method=lambda d: d["method"] + "@GOC")], ignore_index=True).merge(both, on=["campaign", "episode", "q"])
    for campaign, part in pair.loc[np.isclose(pair["q"], q_main) & pair["method"].isin(["B1_chuan@VC", "B1_chuan@GOC"])].groupby("campaign"):
        part = part.loc[part.groupby("episode")["method"].transform("nunique").eq(2)]
        weights_goc = part.loc[part["method"].eq("B1_chuan@GOC"), ["episode", "io_actions_episode"]]
        part = part.drop(columns="io_actions_episode").merge(weights_goc, on="episode")
        for weighting, column in WEIGHTINGS:
            result = run_compare(part, "B1_chuan@VC", "B1_chuan@GOC", cfg, column)
            extra = {f"{key[:-2]}_{side}": value for side in ("VC", "GOC") for key, value in means(part.loc[part["method"].eq(f"B1_chuan@{side}")], "x").items()}
            rows.append(dict(comparison="trước/sau VC: B1_chuan VC − GOC (eligible cả hai)", protocol="VC−GOC", campaign=campaign, q=q_main, weighting=weighting, primary=weighting == "equal", n=result["n"], mean_d=result["mean"], ci95_low=result["lo"], ci95_high=result["hi"], verdict=result["verdict"], **extra))
    return pd.DataFrame(rows)


def io_only(results):
    """Chỉ mốc io_pct (mốc theo ngày dùng cho P5/LOCO, không cho so sánh B1)."""
    return results.loc[results["checkpoint_type"].eq("io_pct")]


def main(argv=None):
    for console in (sys.stdout, sys.stderr):
        console.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None, help="Chạy thử: thư mục gốc chứa P6/ và VC/ (mặc định output_dir trong config).")
    parser.add_argument("--allow-incomplete", action="store_true", help="Chạy thử: bỏ kiểm tra complete và code_dirty của run.json.")
    args = parser.parse_args(argv)
    paths, t7, p6, vc = load_all()
    digest = config_hash(t7, p6, vc, ROOT / paths["t9_output_dir"] / P2_FILE)
    dirs = {"P6": Path(args.output_root) / "P6", "VC": Path(args.output_root) / "VC"} if args.output_root else {"P6": ROOT / p6["output_dir"], "VC": ROOT / vc["output_dir"]}
    for directory in dirs.values():
        check_run(directory, digest, args.allow_incomplete)
    results = pd.concat([read_results(dirs["P6"], "results_*.csv"), read_results(dirs["VC"], "results_*.csv")], ignore_index=True)
    if not results["config_hash"].eq(digest).all() or list(results.columns) != COLUMNS:
        raise SystemExit("Shard có config_hash khác hoặc sai schema.")
    t12_cfg = json.loads((ROOT / paths["t12_config"]).read_text(encoding="utf-8"))
    t12_loco = pd.read_csv(ROOT / paths["t12_output_dir"] / t12_cfg["tiebreak"]["main_outputs"]["loco_choice"])
    outputs = {dirs["P6"] / "p6_dependence.csv": p6_dependence(io_only(results), p6), dirs["P6"] / "p6_control_clusters.csv": p6_control_clusters(dirs["P6"], p6, digest), dirs["VC"] / "p5_vc_all_campaigns.csv": p5_table(results, vc), dirs["VC"] / "vc_loco_choice.csv": loco_table(results, vc, t12_loco), dirs["VC"] / "vc_compare_q25.csv": vc_compare(io_only(results), results, p6, vc)}
    pd.set_option("display.width", 250)
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
        print(f"\n=== {path.name} ({len(frame)} dòng) ===")
        print(frame.head(40).round(3).to_string(index=False))
    print(f"\ncommit {git_state()['commit']} config_hash {digest[:12]}")


if __name__ == "__main__":
    main()
