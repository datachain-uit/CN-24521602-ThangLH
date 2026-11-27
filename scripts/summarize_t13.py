"""T13 bước 5: tóm tắt kết quả DS2-v2 và chấm tiêu chí S3a–c, H1 (Review-2026-11-21 §6.7, §6.8).

Đọc t13_results.csv, table2_v2.csv, t13_meta.csv; ghi summary/*.csv và criteria.json. Mọi ngưỡng đọc từ configs/t13.json.
CI: bootstrap theo data seed bằng compare của T12 (data seed đóng vai "đợt"). --no-criteria: không tính, không in S3a–c, H1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval.protocol_v3 import compare, f1_floor, loco_choice, normalized_gain  # noqa: E402

PATHS = json.loads((ROOT / "configs" / "paths.json").read_text(encoding="utf-8"))
RESULT_REQUIRED = ["campaign", "q", "method", "seed", "n_io", "n_ctrl", "eligible", "f1_star", "nmi", "homogeneity", "wp", "n_clusters_ge2", "runtime_s", "scenario", "mode", "p1_methods", "gamma_shape", "world_mean", "data_seed", "tau_rel_t0", "checkpoint_type", "auc", "tau"]
TABLE2_REQUIRED = ["variant", "data_seed", "community_seed", "pipeline", "configuration", "f1_star", "homogeneity", "weighted_precision", "nmi", "pattern_ok"]
META_REQUIRED = ["world", "data_seed", "scenario", "gamma_shape", "world_mean", "mode", "raised_share_coord_days", "raised_share_all_coord_days", "added_action_share", "fill_share"]
GROUP = ["scenario", "mode", "p1_methods", "gamma_shape", "world_mean"]
START = pd.Timestamp("2020-01-01", tz="UTC")


def load_config():
    return json.loads((ROOT / PATHS["t13_config"]).read_text(encoding="utf-8"))


def require_columns(frame, columns, name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} thiếu cột: {missing}")


def require_seeds(frame, seeds, name):
    """Mỗi nhóm (kịch bản, mode, p1_methods, shape, mean) phải có đúng tập data seed mong đợi."""
    expected = set(int(seed) for seed in seeds)
    for key, part in frame.groupby(GROUP, dropna=False):
        found = set(part["data_seed"].astype(int))
        if found != expected:
            raise ValueError(f"{name} nhóm {dict(zip(GROUP, key))} thiếu seed {sorted(expected - found)}, thừa seed {sorted(found - expected)}.")


def mark_id(row):
    """Mã mốc ổn định giữa các seed: q0.25, pre1440m, day3, full_7d."""
    kind = row["checkpoint_type"]
    if kind == "q":
        return f"q{float(row['q']):.2f}"
    if kind == "pre_t0":
        return f"pre{int(round(-float(row['tau_rel_t0'])))}m"
    if kind == "day":
        return f"day{(pd.Timestamp(row['tau']) + pd.Timedelta(minutes=1) - START).days}"
    return str(kind)


def prepare_results(results, cfg, seeds):
    require_columns(results, RESULT_REQUIRED, "t13_results.csv")
    frame = results.copy()
    frame["data_seed"] = frame["data_seed"].astype(int)
    frame["eligible"] = frame["eligible"].astype(str).str.lower().eq("true")
    for column in ("gamma_shape", "world_mean", "q", "tau_rel_t0", "auc", "f1_star", "n_io", "n_ctrl", "seed"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    require_seeds(frame, seeds, "t13_results.csv")
    frame["mark"] = frame.apply(mark_id, axis=1)
    frame["floor"] = f1_floor(frame["n_io"], frame["n_ctrl"])
    frame["G"] = normalized_gain(frame["f1_star"], frame["floor"])
    return frame


def quantiles(values):
    values = pd.Series(values, dtype=float).dropna()
    return dict(n=len(values), median=values.median(), p10=values.quantile(0.1), p90=values.quantile(0.9))


def paired(a, b, cfg, delta=None):
    """Hiệu theo cặp a − b theo data seed (Series chỉ mục data seed) qua compare của T12: n, mean, lo, hi, verdict."""
    joined = pd.concat({"A": a, "B": b}, axis=1).dropna()
    per = joined.rename_axis("episode").reset_index().melt(id_vars="episode", var_name="method", value_name="f1")
    stats = cfg["stats"]
    return compare(per, "A", "B", delta=stats["delta"] if delta is None else delta, n_boot=stats["n_boot"], seed=stats["boot_seed"])


def series(frame, method, value):
    """Giá trị theo data seed của một method (một dòng mỗi seed)."""
    part = frame.loc[frame["method"].eq(method)]
    if part["data_seed"].duplicated().any():
        raise ValueError(f"{method}: nhiều dòng cho một data seed.")
    return part.set_index("data_seed")[value].astype(float)


def triv_loco(frame):
    """TRIV_LOCO theo data seed trong một nhóm mode × kịch bản × p1_methods × mốc: (điểm, chiều) chọn từ các seed còn lại bằng loco_choice của T12."""
    triv = frame.loc[frame["method"].str.match(r"^TRIV_(first_seen|actions_per_active_day|zero_delta_repost)_(high|low)$")].copy()
    triv["direction"] = triv["method"].str.rsplit("_", n=1).str[1]
    triv["method"] = triv["method"].str.replace(r"^TRIV_", "", regex=True).str.rsplit("_", n=1).str[0]
    triv["campaign"] = triv["data_seed"].astype(str)
    triv["F1_star"] = triv["f1_star"]
    out = {}
    for seed in sorted(triv["data_seed"].unique()):
        score, direction = loco_choice(triv, str(seed))
        own = triv.loc[triv["data_seed"].eq(seed) & triv["method"].eq(score) & triv["direction"].eq(direction), "f1_star"]
        out[seed] = float(own.iloc[0])
    return pd.Series(out, dtype=float)


def scope(frame, cfg, mean):
    """Phạm vi tiêu chí: p1_methods, shape chính, world_mean; B0/B1 chỉ Leiden seed chính."""
    sc = cfg["criteria"]["scope"]
    part = frame.loc[frame["p1_methods"].eq(sc["p1_methods"]) & np.isclose(frame["gamma_shape"], cfg["world"]["gamma_shape"]) & np.isclose(frame["world_mean"], mean)]
    return part.loc[part["seed"].isna() | part["seed"].eq(sc["leiden_seed"])]


def at(frame, scenario, mode, mark):
    return frame.loc[frame["scenario"].eq(scenario) & frame["mode"].eq(mode) & frame["mark"].eq(mark)]


def require_group(frame, scenario, mode, what):
    if not (frame["scenario"].eq(scenario) & frame["mode"].eq(mode)).any():
        raise ValueError(f"Thiếu dữ liệu {what}: kịch bản {scenario}, mode {mode}.")


# --- Tiêu chí ---
def s3a(frame, cfg, scenario):
    c = cfg["criteria"]["s3a"]
    require_group(frame, scenario, c["mode"], "S3a")
    part = frame.loc[frame["scenario"].eq(scenario) & frame["mode"].eq(c["mode"]) & ~frame["checkpoint_type"].isin(c["exclude_checkpoint_types"])]
    lo, hi = c["auc_range"]
    marks = {}
    for mark, rows in part.groupby("mark"):
        entry = {}
        for score in c["scores"]:
            values = rows.loc[rows["method"].eq(f"TRIV_{score}_{c['direction']}") & rows["eligible"], "auc"]
            median = float(values.median()) if len(values) else np.nan
            entry[score] = dict(median_auc=median, n_seeds=int(values.notna().sum()), threshold=[lo, hi], passed=bool(lo <= median <= hi))
        marks[mark] = entry
    q_ref = part.loc[part["checkpoint_type"].eq("q") & np.isclose(part["q"], c["triv_best_before_q"])].groupby("data_seed")["tau_rel_t0"].first()
    best = part.loc[part["method"].eq("TRIV_best")].copy()
    best = best.loc[best["tau_rel_t0"] < best["data_seed"].map(q_ref)]
    triv = {}
    for mark, rows in best.groupby("mark"):
        share = float((rows["f1_star"] >= 1 - c["one_tolerance"]).mean())
        triv[mark] = dict(share_seeds_triv_best_1=share, n_seeds=len(rows), threshold=c["triv_best_one_max_seed_share"], passed=bool(share <= c["triv_best_one_max_seed_share"]))
    volume_ok = all(entry["actions_per_active_day"]["passed"] for entry in marks.values()) if "actions_per_active_day" in c["scores"] else None
    passed = all(e["passed"] for entry in marks.values() for e in entry.values()) and all(e["passed"] for e in triv.values())
    return dict(auc_by_mark=marks, triv_best_before_q=triv, volume_passed=volume_ok, passed=bool(passed))


def s3b_values(frame, cfg, scenario, mode):
    c = cfg["criteria"]["s3b"]
    require_group(frame, scenario, mode, "S3b")
    part = frame.loc[frame["scenario"].eq(scenario) & frame["mode"].eq(mode)]
    pre = part.groupby("mark")["tau_rel_t0"].max()
    out = {}
    for mark in sorted(pre.index[pre < 0]):
        rows = part.loc[part["mark"].eq(mark)]
        g = series(rows, c["method"], "G")
        median = float(g.median())
        entry = dict(g_median=median, n_seeds=int(g.notna().sum()), g_median_max=c["g_median_max"], refs={})
        for ref in c["refs"]:
            result = paired(g, series(rows, ref, "G"), cfg)
            entry["refs"][ref] = dict(mean_diff=result["mean"], ci=[result["lo"], result["hi"]], ci_lo_max=c["ci_lo_max"], passed=bool(result["lo"] <= c["ci_lo_max"]))
        entry["passed"] = bool(median <= c["g_median_max"] and all(r["passed"] for r in entry["refs"].values()))
        out[mark] = entry
    return out


def s3b(frame, cfg, scenario):
    c = cfg["criteria"]["s3b"]
    marks = s3b_values(frame, cfg, scenario, c["mode"])
    report = {mode: s3b_values(frame, cfg, scenario, mode) for mode in c["report_modes"] if (frame["scenario"].eq(scenario) & frame["mode"].eq(mode)).any()}
    for values in report.values():
        for entry in values.values():
            entry.pop("passed")
            for ref in entry["refs"].values():
                ref.pop("passed")
    return dict(by_mark=marks, passed=bool(marks) and all(e["passed"] for e in marks.values()), descriptive=report)


def s3c_value(frame, cfg, scenario, mode, q, method):
    rows = at(frame, scenario, mode, f"q{q:.2f}")
    if rows.empty:
        raise ValueError(f"Thiếu dữ liệu S3c: {scenario}, {mode}, q = {q}.")
    loco = triv_loco(rows)
    result = paired(series(rows, method, "f1_star"), loco, cfg)
    return dict(method=method, ref="TRIV_LOCO", mean_diff=result["mean"], ci=[result["lo"], result["hi"]], delta=cfg["stats"]["delta"], verdict=result["verdict"], b1_median=float(series(rows, method, "f1_star").median()), loco_median=float(loco.median()))


def s3c(frame, cfg):
    c = cfg["criteria"]["s3c"]
    out = {}
    for scenario in c["scenarios"]:
        main = s3c_value(frame, cfg, scenario, c["mode"], c["q"], c["method"])
        main["passed"] = main["verdict"] == c["verdict_required"]
        report = [s3c_value(frame, cfg, scenario, c["mode"], q, method) for q in [c["q"], *c["report_q"]] for method in [c["method"], *c["report_methods"]] if not (q == c["q"] and method == c["method"])]
        out[scenario] = dict(main=main, report=report, passed=main["passed"])
    for scenario in c["report_scenarios"]:
        if (frame["scenario"].eq(scenario)).any():
            out[scenario] = dict(report=[s3c_value(frame, cfg, scenario, c["mode"], q, method) for q in [c["q"], *c["report_q"]] for method in [c["method"], *c["report_methods"]]], passed=None)
    return out


def h1_value(frame, cfg, scenario, q):
    c = cfg["criteria"]["h1"]
    method = f"TRIV_{c['score']}_{c['direction']}"
    clean, seckin = (at(frame, scenario, mode, f"q{q:.2f}") for mode in (c["clean_mode"], c["seckin_mode"]))
    if clean.empty or seckin.empty:
        raise ValueError(f"Thiếu dữ liệu H1: {scenario}, q = {q}.")
    a, b = series(seckin, method, "auc"), series(clean, method, "auc")
    result = paired(a, b, cfg)
    lo, hi = c["clean_auc_range"]
    entry = dict(clean_median_auc=float(b.median()), seckin_median_auc=float(a.median()), paired_mean_diff=result["mean"], ci=[result["lo"], result["hi"]], clean_auc_range=[lo, hi], seckin_auc_min=c["seckin_auc_min"], paired_ci_lo_gt=c["paired_ci_lo_gt"])
    entry["checks"] = dict(clean_in_range=bool(lo <= entry["clean_median_auc"] <= hi), seckin_high=bool(entry["seckin_median_auc"] >= c["seckin_auc_min"]), paired_ci_lo=bool(result["lo"] > c["paired_ci_lo_gt"]))
    entry["passed"] = all(entry["checks"].values())
    return entry


def h1(frame, cfg, scenarios):
    c = cfg["criteria"]["h1"]
    out = {scenario: dict(main=h1_value(frame, cfg, scenario, c["q"]), report={f"q{q:.2f}": h1_value(frame, cfg, scenario, q) for q in c["report_q"]}) for scenario in scenarios}
    n_pass = sum(out[s]["main"]["passed"] for s in scenarios)
    verdict = c["aggregate"]["all"] if n_pass == len(scenarios) else c["aggregate"]["some"] if n_pass else c["aggregate"]["none"]
    return dict(by_scenario=out, n_scenarios_passed=n_pass, aggregate=verdict)


def evaluate_criteria(frame, cfg, n_seeds):
    """criteria.json cho mỗi world_mean có trong kết quả; mỗi tiêu chí có giá trị, CI, ngưỡng, đạt/không đạt."""
    scenarios = [f"sim{sim}" for sim in cfg["scenarios"]]
    out = {}
    for mean in sorted(frame["world_mean"].dropna().unique()):
        part = scope(frame, cfg, mean)
        s3a_result = {scenario: s3a(part, cfg, scenario) for scenario in scenarios}
        volume_failed = any(r["volume_passed"] is False for r in s3a_result.values())
        out[f"world_mean={mean:g}"] = dict(
            S3a=s3a_result, S3b={scenario: s3b(part, cfg, scenario) for scenario in scenarios}, S3c=s3c(part, cfg), H1=h1(part, cfg, scenarios),
            fallback=dict(rule=cfg["fallback_rule"], s3a_volume_failed=volume_failed, n_seeds=n_seeds, eligible_to_trigger=bool(volume_failed and n_seeds == len(cfg["data_seeds"]) and float(mean) == float(cfg["world"]["mean_actions_per_day"]))),
        )
    return out


# --- Bảng tóm tắt ---
def table2_summary(table2, v1):
    """Bảng 2 v2 (hai biến thể) so với v1 (T6b): trung vị, P10–P90 của F1*, H, WP, NMI (v1 không có NMI) và tỷ lệ đúng mẫu."""
    require_columns(table2, TABLE2_REQUIRED, "table2_v2.csv")
    frames = [table2.assign(version="v2_" + table2["variant"].astype(str))]
    if v1 is not None:
        v1 = v1.copy()
        v1["configuration"] = v1["configuration"].astype(str)
        keys = np.where(v1["configuration"].eq("3"), v1[["homogeneity", "weighted_precision"]].min(axis=1), v1[["f1_star", "homogeneity", "weighted_precision"]].min(axis=1))
        frames.append(v1.assign(version="v1", nmi=np.nan, pattern_ok=keys >= 1 - 1e-9))
    data = pd.concat(frames, ignore_index=True)
    data["pattern_ok"] = data["pattern_ok"].astype(str).str.lower().eq("true")
    rows = []
    for (version, pipeline, configuration), part in data.groupby(["version", "pipeline", data["configuration"].astype(str)]):
        row = dict(version=version, pipeline=pipeline, configuration=configuration, n_runs=len(part), pattern_rate=float(part["pattern_ok"].mean()))
        for metric in ("f1_star", "homogeneity", "weighted_precision", "nmi"):
            stats = quantiles(pd.to_numeric(part[metric], errors="coerce"))
            row.update({f"{metric}_{key}": value for key, value in stats.items() if key != "n"})
        rows.append(row)
    return pd.DataFrame(rows)


def auc_summary(frame):
    """AUC và F1* của từng điểm tầm thường theo mode × kịch bản × p1 × shape × mean × mốc × chiều (chỉ dòng eligible)."""
    triv = frame.loc[frame["method"].str.startswith("TRIV_") & frame["eligible"]]
    rows = []
    for key, part in triv.groupby([*GROUP, "mark", "method"]):
        auc, f1 = quantiles(part["auc"]), quantiles(part["f1_star"])
        rows.append(dict(zip([*GROUP, "mark", "method"], key), n_seeds=auc["n"], auc_median=auc["median"], auc_p10=auc["p10"], auc_p90=auc["p90"], f1_median=f1["median"], f1_p10=f1["p10"], f1_p90=f1["p90"]))
    return pd.DataFrame(rows)


def model_summary(frame, leiden_seed):
    """F1*, sàn, G, NMI, H, WP, số cụm >= 2 và runtime của B0/B1 (Leiden seed chính) theo mode × kịch bản × p1 × mốc."""
    models = frame.loc[frame["method"].str.match(r"^(B0|B1)_") & frame["seed"].eq(leiden_seed)]
    rows = []
    for key, part in models.groupby([*GROUP, "mark", "method"]):
        row = dict(zip([*GROUP, "mark", "method"], key), n_seeds=len(part), eligible_share=float(part["eligible"].mean()), n_io_median=part["n_io"].median(), n_ctrl_median=part["n_ctrl"].median())
        for metric in ("f1_star", "floor", "G", "nmi", "homogeneity", "wp", "n_clusters_ge2", "runtime_s"):
            stats = quantiles(pd.to_numeric(part[metric], errors="coerce"))
            row.update({f"{metric}_{name}": value for name, value in stats.items() if name != "n"})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(frame, cfg):
    """Hiệu theo cặp (mode − clean) theo data seed kèm CI bootstrap: AUC khối lượng (chiều cố định), F1* và G của B1-chuẩn/B1-gốc."""
    volume = f"TRIV_{cfg['volume_score']['score']}_{cfg['volume_score']['direction']}"
    leiden = cfg["criteria"]["scope"]["leiden_seed"]
    rows = []
    keys = ["scenario", "p1_methods", "gamma_shape", "world_mean", "mark"]
    for key, part in frame.groupby(keys):
        clean = part.loc[part["mode"].eq("clean")]
        if clean.empty:
            continue
        for mode in sorted(set(part["mode"]) - {"clean"}):
            other = part.loc[part["mode"].eq(mode)]
            for metric, method, value in ((f"auc_{volume}", volume, "auc"), ("f1_B1_chuan", "B1_chuan", "f1_star"), ("G_B1_chuan", "B1_chuan", "G"), ("f1_B1_goc", "B1_goc", "f1_star"), ("G_B1_goc", "B1_goc", "G")):
                pick = (lambda f: f.loc[f["seed"].eq(leiden)]) if method.startswith("B1") else (lambda f: f)
                a, b = series(pick(other), method, value), series(pick(clean), method, value)
                if a.dropna().empty or b.dropna().empty:
                    continue
                result = paired(a, b, cfg)
                rows.append(dict(zip(keys, key), mode=mode, ref_mode="clean", metric=metric, n=result["n"], mean_diff=result["mean"], median_diff=float((a - b).dropna().median()), ci_lo=result["lo"], ci_hi=result["hi"]))
    return pd.DataFrame(rows)


def raised_summary(meta):
    """Tỷ lệ bị nâng (i), (ii), (iii), tỷ lệ bổ sung control và Δt = 0 theo nhãn: trung vị và lớn nhất theo thế giới × shape × mean × kịch bản × mode."""
    require_columns(meta, META_REQUIRED, "t13_meta.csv")
    columns = [column for column in meta.columns if column.startswith(("raised_share", "added_action_share", "fill_share", "dz_"))]
    data = meta.copy()
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    keys = ["world", "gamma_shape", "world_mean", "scenario", "mode"]
    grouped = data.groupby(keys, dropna=False)[columns]
    return grouped.median().add_suffix("_median").join(grouped.max().add_suffix("_max")).join(data.groupby(keys, dropna=False).size().rename("n_seeds")).reset_index()


def shape_summary(frame, cfg):
    """Độ nhạy shape: AUC khối lượng và xuất hiện đầu tiên (chiều thấp) ở q chính, mọi mode, p1_methods = off."""
    methods = [f"TRIV_{cfg['volume_score']['score']}_{cfg['volume_score']['direction']}", f"TRIV_{cfg['first_seen_score']['score']}_{cfg['first_seen_score']['direction']}"]
    part = frame.loc[frame["p1_methods"].eq("off") & frame["mark"].eq(f"q{cfg['marks']['q_main']:.2f}") & frame["method"].isin(methods) & frame["eligible"]]
    rows = []
    for key, rows_part in part.groupby(["world_mean", "gamma_shape", "scenario", "mode", "method"]):
        stats = quantiles(rows_part["auc"])
        rows.append(dict(zip(["world_mean", "gamma_shape", "scenario", "mode", "method"], key), n_seeds=stats["n"], auc_median=stats["median"], auc_p10=stats["p10"], auc_p90=stats["p90"]))
    return pd.DataFrame(rows)


def leiden_check(frame, cfg):
    """Tính gần tất định của Leiden: trên các seed có đủ leiden_seeds_check, tỷ lệ cặp (data seed, mốc) có độ lệch chuẩn F1* giữa seed Leiden = 0."""
    seeds = cfg["leiden_seeds_check"]
    models = frame.loc[frame["method"].str.startswith("B1_") & frame["seed"].isin(seeds)]
    counts = models.groupby([*GROUP, "data_seed", "mark", "method"])["seed"].nunique()
    full = counts[counts.eq(len(seeds))].index
    if not len(full):
        return pd.DataFrame(columns=[*GROUP, "method", "n_pairs", "share_std_zero"])
    std = models.set_index([*GROUP, "data_seed", "mark", "method"]).loc[full].groupby(level=list(range(len(GROUP) + 3)))["f1_star"].std(ddof=0)
    table = std.reset_index().assign(zero=lambda f: f["f1_star"].abs().le(1e-12))
    return table.groupby([*GROUP, "method"]).agg(n_pairs=("zero", "size"), share_std_zero=("zero", "mean")).reset_index()


def main(argv=None):
    for console in (sys.stdout, sys.stderr):
        console.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=None, help="Thư mục kết quả của run_t13; mặc định t13_output_dir.")
    parser.add_argument("--n-data-seeds", type=int, default=None, help="Số data seed đầu mong đợi (thử khói); mặc định đủ data_seeds.")
    parser.add_argument("--no-criteria", action="store_true", help="Không tính và không in S3a–c, H1.")
    args = parser.parse_args(argv)
    cfg = load_config()
    in_dir = Path(args.input_dir) if args.input_dir else ROOT / PATHS["t13_output_dir"]
    out_dir = in_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = cfg["data_seeds"][: args.n_data_seeds] if args.n_data_seeds else cfg["data_seeds"]
    frame = prepare_results(pd.read_csv(in_dir / "t13_results.csv"), cfg, seeds)
    v1_path = ROOT / PATHS["t6b_full_dir"] / "results_all.csv"
    tables = {
        "table2_v2_vs_v1.csv": table2_summary(pd.read_csv(in_dir / "table2_v2.csv"), pd.read_csv(v1_path) if v1_path.is_file() else None),
        "auc_by_mode.csv": auc_summary(frame),
        "models_by_mode_mark.csv": model_summary(frame, cfg["criteria"]["scope"]["leiden_seed"]),
        "paired_diffs.csv": paired_summary(frame, cfg),
        "raised_shares.csv": raised_summary(pd.read_csv(in_dir / "t13_meta.csv")),
        "shape_sensitivity.csv": shape_summary(frame, cfg),
        "leiden_check.csv": leiden_check(frame, cfg),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / name, index=False)
        print(f"Đã ghi {out_dir / name} ({len(table)} dòng).", flush=True)
    if args.no_criteria:
        print("--no-criteria: bỏ qua S3a–c, H1.", flush=True)
        return
    result = evaluate_criteria(frame, cfg, len(seeds))
    (in_dir / "criteria.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    for mean, block in result.items():
        print(f"[{mean}] S3a: {{{', '.join(f'{s}: {r['passed']}' for s, r in block['S3a'].items())}}}")
        print(f"[{mean}] S3b: {{{', '.join(f'{s}: {r['passed']}' for s, r in block['S3b'].items())}}}")
        print(f"[{mean}] S3c: {{{', '.join(f'{s}: {r['passed']}' for s, r in block['S3c'].items())}}}")
        print(f"[{mean}] H1: {block['H1']['aggregate']} ({block['H1']['n_scenarios_passed']}/{len(block['H1']['by_scenario'])} kịch bản)")
    print(f"Đã ghi {in_dir / 'criteria.json'}.", flush=True)


if __name__ == "__main__":
    main()
