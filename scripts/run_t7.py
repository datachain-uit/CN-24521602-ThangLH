"""T7: chạy khung đánh giá theo mốc (batch theo checkpoint, Review 15/11 Mục 6.6) và ghi reports/T7/t7_results.csv (append) cùng run.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.seckin_protocol import INPUT_COLUMNS, MODEL_COLUMNS, EpisodeRules, build_events, daily_coverage, harmonize_time, make_checkpoints, make_episodes  # noqa: E402
from src.eval.t7_checkpoints import B1_VARIANTS, RESULT_COLUMNS, append_result, ds2_job, ds2_trivial_rows, format_tau, iter_campaign_rows, load_config, read_done_keys, result_key, trivial_rows  # noqa: E402
from scripts.run_t6b_experiment import generate_data  # noqa: E402  (D2-c: dữ liệu DS2 sinh như T6b, làm tròn giây)

PACKAGES = ["numpy", "pandas", "networkx", "igraph", "leidenalg", "scipy", "scikit-learn", "time-aware-collaboration"]
NOTES = {
    "scope": "Batch theo checkpoint, không phải cập nhật tăng dần (C1 bắt đầu sau 30/11).",
    "layer_weights": "Phương án (c): đồ thị Brent/Leiden giữ ngưỡng edge_tolerance; layer_weights của Surprise = edges_author = số cặp tài khoản có chung nội dung trong lớp, đếm như get_time_damped_edge_list (models.py:283-315), giống nhau cho B1-gốc và B1-chuẩn.",
    "zero_only_pairs": "Cặp tài khoản chỉ có Δt = 0: tác giả vẫn sinh cạnh trọng số 0 (ở B1-gốc); ta không tạo cạnh nhưng vẫn đếm cặp đó trong layer_weights.",
    "beta": "beta chọn một lần mỗi (tau, lớp, biến thể) bằng Brent bounded như calculate_optimal_beta, Louvain networkx với random.Random(community_seeds[0]); 5 seed chỉ dùng cho Leiden.",
    "runtime_s": "B1: ghép cặp tại tau chia đều cho 2 biến thể × số seed + (Brent + dựng đồ thị của biến thể) / số seed + Leiden của seed. B0: riêng từng seed. TRIV: thời gian 3 điểm số chia đều cho 7 dòng.",
    "b0": "B0 = nhãn ngẫu nhiên đều K ∈ {2, 3} trên tập đánh giá. Với WP chuẩn, E[WP] = π + (1 − π)(K − 1)/(N − 1), π = N_IO/N (expected_random_wp); 0,52/0,36 của random_labeler tác giả là WP với p_k = recall của control (S2a).",
    "s3_ds2": "Trên DS2, tài khoản phối hợp (coord_XX) không có sự kiện trước t0 = 1920 ở mọi seed, nên S3 là test chống rò rỉ tương lai, không kiểm khả năng phát hiện.",
    "ds2_l3": "DS2 cấu hình L3 dùng find_partition_multiplex 1 lớp (Surprise), khác greedy_modularity của nhánh gốc T6b, nên không so trực tiếp với F1* = 0,533.",
    "triv_zero_delta_ds2": "Trên DS2 (thời gian làm tròn giây, 0 cặp Δt = 0) điểm tỷ lệ Δt = 0 là hằng số; F1* tầm thường của điểm này chỉ còn phụ thuộc thứ tự phá hòa ngẫu nhiên (triv_tiebreak) và luôn >= 2N_IO/(N + N_IO) (k = N, gán tất cả là IO). Trước khi đổi bí danh ID, nó bằng đúng 2N_IO/(N + N_IO) vì auth_XX < coord_XX.",
    "triv_tiebreak": "TRIV: trước khi gọi oracle_top_k (T9, không sửa), accountid được đổi thành bí danh theo hoán vị ngẫu nhiên với seed triv_tiebreak_seed, nên phá hòa không tương quan với nhãn qua thứ tự ID (trên DS2 auth_XX < coord_XX). Áp dụng cho cả DS1 và DS2; B0/B1 không đổi.",
    "ds2_dedup_scores": "DS2: điểm (i) thời điểm xuất hiện và (ii) hành động/ngày bỏ sự kiện trùng hoàn toàn giữa các lớp (cùng tài khoản, thời điểm, action_type, content_id) vì phần authentic giống hệt nhau ở 3 lớp; điểm (iii) và đồ thị B1 giữ nguyên như T6b.",
    "degenerate_no_io": "Khi N_IO = 0 (mốc trước t0 của DS2), sklearn và mã tác giả cho homogeneity = nmi = nmi50 = 1 (nhãn đúng chỉ có một lớp); giữ nguyên để nhất quán với U3. S3 chỉ đọc qua count_true_detections và F1*; khi tổng hợp luôn lọc eligible=True.",
    "ineligible": "Mốc không đạt (≥ 5 IO và ≥ 5 control): mỗi method một dòng, eligible=False, chỉ số NaN, không chạy mô hình. Mốc trước t0 của DS2 vẫn chạy B0/B1, TRIV ghi NaN.",
}


def campaign_files(source, campaign):
    """Chép từ notebooks/T9_EDA_Seckin.ipynb (ô 3): các shard Parquet của một chiến dịch theo thứ tự phần."""
    pattern = re.compile(rf"{re.escape(campaign)}(?:_part_(\d+))?(?:\.gzip)?\.parquet", re.IGNORECASE)
    shards = {}
    for path in source.glob("*.parquet"):
        match = pattern.fullmatch(path.name)
        if match:
            part = int(match[1]) if match[1] else None
            if part in shards:
                raise ValueError(f"{campaign} có shard trùng thứ tự.")
            shards[part] = path
    if not shards:
        raise FileNotFoundError(f"Thiếu Parquet cho {campaign} trong seckin_dir.")
    if None in shards and len(shards) > 1:
        raise ValueError(f"{campaign} có đồng thời tệp nguyên và các phần.")
    return [shards[i] for i in sorted(shards)]


def load_ds1(campaign, cfg, paths):
    """P1–P4 bằng hàm T9; đối chiếu mốc với reports/T9 p4_checkpoints.csv; trả luồng và mốc q (io_pct)."""
    rules = EpisodeRules()
    frame = harmonize_time(pd.concat([pd.read_parquet(path, columns=INPUT_COLUMNS) for path in campaign_files(ROOT / paths["seckin_dir"], campaign)], ignore_index=True))
    stream, episodes = make_episodes(frame, daily_coverage(frame, rules), rules, cfg["control_min"])
    checkpoints = make_checkpoints(stream, episodes, rules)
    reference = pd.read_csv(ROOT / paths["t9_output_dir"] / "p4_checkpoints.csv")
    reference = reference.loc[reference["campaign"].eq(campaign)].reset_index(drop=True)
    ours = checkpoints.reset_index(drop=True)
    same = len(ours) == len(reference) and ours[["episode", "n_io", "n_ctrl", "eligible"]].astype(str).equals(reference[["episode", "n_io", "n_ctrl", "eligible"]].astype(str)) and ours["tau"].map(format_tau).equals(reference["tau"].map(format_tau))
    if not same:
        raise ValueError(f"Mốc của {campaign} không khớp p4_checkpoints.csv của T9.")
    marks = checkpoints.loc[checkpoints["checkpoint_type"].eq("io_pct"), ["episode", "q", "tau", "n_io", "n_ctrl", "eligible"]]
    if sorted(marks["q"].unique()) != sorted(cfg["q_values"]):
        raise ValueError("q của T9 khác q_values trong configs/t7.json.")
    return stream, marks.reset_index(drop=True)


def ordered_marks(marks, campaign, cfg):
    """Mốc q ưu tiên (priority_q) của chiến dịch chạy trước, rồi các mốc còn lại; giữ thứ tự đợt."""
    priority = cfg["priority_q"].get(campaign)
    if priority is None:
        return marks
    first = np.isclose(marks["q"].astype(float), priority)
    return pd.concat([marks.loc[first], marks.loc[~first]], ignore_index=True)


CODE_DIRS = ["src", "scripts", "configs", "tests"]


def git_info():
    """git hash, cờ dirty, sha256 của git diff HEAD và sha256 từng file untracked trong src/ scripts/ configs/ tests/ để truy vết code khi chưa commit."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or None
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE_DIRS], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    diff = subprocess.run(["git", "diff", "HEAD", "--binary", "--", *CODE_DIRS], cwd=ROOT, capture_output=True, check=False).stdout
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", *CODE_DIRS], cwd=ROOT, capture_output=True, text=True, check=False).stdout.split()
    return dict(git_head=head, code_dirty=bool(dirty), git_diff_sha256=hashlib.sha256(diff).hexdigest(), untracked_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sorted(untracked)})


def package_versions():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def beta_bound_table(csv_path, cfg):
    """Đếm beta chạm biên trên (>= bounds[1] − xatol, "β ≥ 10") và biên dưới (<= bounds[0] + xatol, "β = 0") theo campaign × lớp × biến thể từ CSV; mỗi (campaign, episode, tau, method) tính một lần."""
    if not Path(csv_path).is_file():
        return []
    frame = pd.read_csv(csv_path)
    frame = frame.loc[frame["method"].isin([name for name, _ in B1_VARIANTS])].drop_duplicates(["campaign", "episode", "tau", "method"])
    upper, lower = cfg["beta_bounds"][1] - cfg["brent_xatol"], cfg["beta_bounds"][0] + cfg["brent_xatol"]
    rows = []
    for (campaign, method), part in frame.groupby(["campaign", "method"]):
        for column in [c for c in RESULT_COLUMNS if c.startswith("beta_")]:
            selected = part[column].dropna()
            if len(selected):
                rows.append(dict(campaign=campaign, layer=column[5:], variant=method, n_selected=int(len(selected)), n_at_upper=int(selected.ge(upper).sum()), n_at_lower=int(selected.le(lower).sum())))
    return rows


def ds2_ineligible_table(csv_path):
    """Số mốc q của DS2 và số mốc không đạt theo cấu hình × q (mỗi (campaign, tau) tính một lần; không tính mốc trước t0)."""
    if not Path(csv_path).is_file():
        return []
    frame = pd.read_csv(csv_path)
    marks = frame.loc[frame["campaign"].str.startswith("DS2_") & frame["q"].notna()].drop_duplicates(["campaign", "tau"])
    marks = marks.assign(configuration=marks["campaign"].str.split("_").str[1])
    table = marks.groupby(["configuration", "q"]).agg(n_marks=("eligible", "size"), n_ineligible=("eligible", lambda values: int((~values.astype(bool)).sum()))).reset_index()
    return table.to_dict("records")


def write_run_json(path, cfg, args, runtimes, csv_path, started_utc):
    """Ghi đè run.json (ghi file tạm rồi đổi tên) sau mỗi chiến dịch; runtime theo chiến dịch cộng dồn qua các lần chạy lại."""
    path = Path(path)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    merged = dict(previous.get("runtime_s_by_campaign", {}))
    merged.update({campaign: round(merged.get(campaign, 0.0) + seconds, 1) for campaign, seconds in runtimes.items()})
    payload = dict(task="T7", review="2026-11-15 §6.6", started_utc=started_utc, updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), **git_info(), packages=package_versions(), args=vars(args), config=cfg, runtime_s_by_campaign=merged, beta_at_bound=beta_bound_table(csv_path, cfg), ds2_ineligible_marks=ds2_ineligible_table(csv_path), notes=NOTES, results_csv=Path(csv_path).name)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def load_ds2(cfg):
    """DS2: mỗi data seed × cấu hình (L123, L3) là một chiến dịch 1 đợt, tên DS2_{cấu hình}_d{seed}."""
    min_accounts = EpisodeRules().min_accounts
    jobs = []
    for data_seed in cfg["ds2"]["data_seeds"]:
        datasets, truth, _ = generate_data(data_seed)
        for configuration, sims in cfg["ds2"]["configurations"].items():
            jobs.append((*ds2_job(datasets, truth, sims, f"DS2_{configuration}_d{data_seed}", cfg, min_accounts), ds2_trivial_rows))
    return jobs


def parse_args(argv, cfg):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaigns", nargs="+", default=cfg["run_order"], help="Chiến dịch theo thứ tự chạy, mặc định run_order trong configs/t7.json.")
    parser.add_argument("--max-checkpoints", type=int, default=None, help="Giới hạn số mốc mỗi chiến dịch (sau khi sắp ưu tiên).")
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["community_seeds"], help="Seed Leiden và B0, mặc định community_seeds.")
    parser.add_argument("--save-partitions", action="store_true", help="Lưu phân hoạch B0/B1 (toàn bộ nút được phân hoạch, cột in_eval) ra parquet trong partitions_dir của configs/paths.json (thư mục bị .gitignore chặn); mặc định tắt.")
    parser.add_argument("--output-dir", default=None, help="Thư mục ghi CSV và run.json (chạy thử); mặc định t7_output_dir trong configs/paths.json.")
    return parser.parse_args(argv)


PARTITION_COLUMNS = ["campaign", "episode", "q", "tau", "method", "seed", "accountid", "cluster", "in_eval"]


def partition_path(directory, base):
    """Một parquet mỗi (campaign, episode, q); tau thêm vào tên để các mốc trước t0 của DS2 (q = NaN) không trùng."""
    _, episode, q, tau, _, _ = result_key({**base, "method": "", "seed": float("nan")})
    return Path(directory) / str(base["campaign"]) / f"episode{episode}_q{q}_tau{tau.replace(':', '')}.parquet"


def check_ignored(directory):
    """Dừng nếu partitions_dir không bị .gitignore chặn (git check-ignore, chỉ đọc)."""
    probe = Path(directory) / "probe.parquet"
    ignored = subprocess.run(["git", "check-ignore", "-q", str(probe.relative_to(ROOT) if probe.is_relative_to(ROOT) else probe)], cwd=ROOT, capture_output=True, check=False).returncode == 0
    if not ignored:
        raise RuntimeError("partitions_dir không bị .gitignore chặn; không lưu phân hoạch.")


def partition_writer(directory):
    """Trả partition_fn cho iter_campaign_rows: ghi parquet của một mốc; nếu tệp đã có (resume) thì gộp, dòng mới thay dòng cũ cùng (method, seed, accountid). Không in accountid."""
    def write(base, records):
        path = partition_path(directory, base)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(records).assign(campaign=base["campaign"], episode=int(base["episode"]), q=float(base["q"]), tau=format_tau(base["tau"]))
        frame = frame.assign(seed=frame["seed"].astype(int), accountid=frame["accountid"].astype(str), cluster=frame["cluster"].astype(int), in_eval=frame["in_eval"].astype(bool))[PARTITION_COLUMNS]
        if path.is_file():
            frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True).drop_duplicates(["method", "seed", "accountid"], keep="last")
        frame.to_parquet(path, index=False)
    return write


def resolve_output_dir(args, paths):
    """--output-dir nếu có, ngược lại ROOT / t7_output_dir."""
    return Path(args.output_dir) if args.output_dir else ROOT / paths["t7_output_dir"]


def main(argv=None):
    for console in (sys.stdout, sys.stderr):
        console.reconfigure(encoding="utf-8")
    cfg = load_config()
    args = parse_args(argv, cfg)
    paths = json.loads((ROOT / "configs" / "paths.json").read_text(encoding="utf-8"))
    out_dir = resolve_output_dir(args, paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, run_path = out_dir / "t7_results.csv", out_dir / "run.json"
    done = read_done_keys(csv_path)
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Kết quả: {csv_path} ({len(done)} dòng đã có).", flush=True)
    partition_fn = None
    if args.save_partitions:
        partitions_dir = ROOT / paths["partitions_dir"]
        check_ignored(partitions_dir)
        partition_fn = partition_writer(partitions_dir)
        print(f"Lưu phân hoạch vào {paths['partitions_dir']}.", flush=True)
    for campaign in args.campaigns:
        clock = perf_counter()
        if campaign == "DS2":
            jobs = load_ds2(cfg)
        else:
            stream, marks = load_ds1(campaign, cfg, paths)
            jobs = [(campaign, stream, ordered_marks(marks, campaign, cfg), lambda history: build_events(history[MODEL_COLUMNS], tuple(cfg["ds1_layers"])), cfg["ds1_layers"], trivial_rows)]
        written = 0
        for name, stream, marks, events_fn, layers, trivial_fn in jobs:
            marks = marks if args.max_checkpoints is None else marks.head(args.max_checkpoints)
            for row in iter_campaign_rows(stream, marks, events_fn, name, cfg, layers, args.seeds, done=done, trivial_fn=trivial_fn, progress=lambda text: print(text, flush=True), partition_fn=partition_fn):
                append_result(csv_path, row)
                done.add(result_key(row))
                written += 1
        runtime = perf_counter() - clock
        print(f"[{campaign}] xong: {written} dòng mới, {runtime:.1f}s.", flush=True)
        write_run_json(run_path, cfg, args, {campaign: round(runtime, 1)}, csv_path, started_utc)


if __name__ == "__main__":
    main()
