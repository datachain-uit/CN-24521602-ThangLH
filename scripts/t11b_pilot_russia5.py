"""T11b: kiểm tra độ phủ tệp thử Russia_5_part_1 bằng nguyên các hàm T9 (Review-2026-11-22 §6.2) và ghi reports/T11/pilot_Russia_5_part_1.json. Chỉ xuất thống kê tổng hợp, không có accountid hay nội dung bài đăng."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.seckin_protocol import EpisodeRules, INPUT_COLUMNS, daily_coverage, harmonize_time, make_episodes  # noqa: E402

FILE_NAME = "Russia_5_part_1.gzip.parquet"
OUTPUT_NAME = "pilot_Russia_5_part_1.json"
CONTROL_MIN = 1
THRESHOLD = 0.20  # Ngưỡng của quy tắc T11b (Review-2026-11-22 §6.2), so sánh >=, không làm tròn.
RULE_TEXT = {
    "tai_tiep": "share_all >= 0.20 và qualified >= 1",
    "bao_GVHD": "share_all < 0.20 nhưng share_after_first_control >= 0.20",
    "dung": "mọi trường hợp còn lại",
}


def decide(share_all, share_after_first_control, qualified):
    """Quy tắc chốt trước (Review-2026-11-22 §6.2). Giá trị None (không tính được) rơi vào nhánh 'dung'."""
    if share_all is None:
        return "dung"
    if share_all >= THRESHOLD and qualified >= 1:
        return "tai_tiep"
    if share_all < THRESHOLD and share_after_first_control is not None and share_after_first_control >= THRESHOLD:
        return "bao_GVHD"
    return "dung"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def day_str(value):
    return None if pd.isna(value) else str(value.date())


def compute_pilot(frame, rules):
    """Đúng mã tham chiếu §6.2; thêm xử lý io_total == 0, both rỗng và khóa 'extra' chỉ gồm số đếm."""
    daily = daily_coverage(frame, rules)
    covered = bool(daily[f"covered_c{CONTROL_MIN}"].any())
    io_total = int(daily["io_actions"].sum())
    first_ctrl = daily.index[daily["ctrl_accounts"].gt(0)].min()
    io_before = int(daily.loc[daily.index < first_ctrl, "io_actions"].sum()) if pd.notna(first_ctrl) else None
    both = daily.loc[daily["io_accounts"].gt(0) & daily["ctrl_accounts"].gt(0)]
    diff = both["ctrl_accounts"] - both["io_accounts"]
    modes = diff.mode()
    mode = int(modes.iloc[0]) if len(modes) else None
    extra = dict(
        control_min=CONTROL_MIN, n_rows=len(frame), first_control_day=day_str(first_ctrl),
        days_observed=len(daily), days_with_io=int(daily["io_accounts"].gt(0).sum()), days_with_control=int(daily["ctrl_accounts"].gt(0).sum()),
        days_with_both=len(both), days_covered_c1=int(daily[f"covered_c{CONTROL_MIN}"].sum()),
        ctrl_minus_io_modes=[int(m) for m in modes], ctrl_minus_io_mode_days=int(diff.eq(mode).sum()) if mode is not None else None,
        ctrl_minus_io_mode_pct_days=100 * float(diff.eq(mode).mean()) if mode is not None else None,
        ctrl_minus_io_mode_null_reason=None if mode is not None else "không có ngày nào có cả IO và control",
        io_unique_accounts=int(frame.loc[~frame["is_control"], "accountid"].nunique()), ctrl_unique_accounts=int(frame.loc[frame["is_control"], "accountid"].nunique()),
        p1_excluded_accounts=None, p1_excluded_accounts_note="harmonize_time không loại tài khoản mà dừng (ValueError) nếu có tài khoản mang cả hai nhãn; chạy qua được nghĩa là 0 tài khoản như vậy.",
    )
    pilot = dict(first_day=day_str(daily.index.min()), last_day=day_str(daily.index.max()), io_actions=io_total, io_before_first_control=io_before)
    if not covered:
        pilot.update(episodes=0, qualified=0, share_all=None, share_after_first_control=None, ctrl_minus_io_mode=mode)
        return pilot, extra, "không có ngày được phủ"
    stream, episodes = make_episodes(frame, daily, rules, control_min=CONTROL_MIN)
    io_qualified = int(episodes.loc[episodes["qualified"], "io_actions"].sum())
    after = io_total - (io_before or 0)
    pilot.update(
        episodes=len(episodes), qualified=int(episodes["qualified"].sum()),
        share_all=io_qualified / io_total if io_total else None,
        share_after_first_control=io_qualified / max(after, 1) if io_total else None, ctrl_minus_io_mode=mode,
    )
    extra.update(io_actions_qualified=io_qualified, io_actions_in_stream=int(episodes["io_actions"].sum()), episodes_table=[
        dict(episode=int(ep), start=day_str(row["start"]), end=day_str(row["end"]), covered_days=int(row["covered_days"]), span_days=int(row["span_days"]),
             io_accounts=int(row["io_accounts"]), ctrl_accounts=int(row["ctrl_accounts"]), io_actions=int(row["io_actions"]), ctrl_actions=int(row["ctrl_actions"]), qualified=bool(row["qualified"]))
        for ep, row in episodes.iterrows()
    ])
    return pilot, extra, None if io_total else "io_total == 0"


def main():
    from scripts.run_t7 import git_info  # Cùng định nghĩa git_head/code_dirty như run.json của T7.
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")  # Console Windows cp1252 không in được tiếng Việt.
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    git = git_info()
    if git["code_dirty"]:
        raise SystemExit("Cây mã (src/ scripts/ configs/ tests/) chưa sạch: commit trước khi chạy T11b.")
    paths = json.loads((ROOT / "configs/paths.json").read_text(encoding="utf-8"))
    source = ROOT / paths["seckin_dir"] / FILE_NAME
    out_path = ROOT / paths["t11_output_dir"] / OUTPUT_NAME
    rules = EpisodeRules()
    raw = pd.read_parquet(source, columns=INPUT_COLUMNS)
    raw_id_dtype = str(raw["accountid"].dtype)
    frame = harmonize_time(raw)
    pilot, extra, stop_reason = compute_pilot(frame, rules)
    decision = decide(pilot["share_all"], pilot["share_after_first_control"], pilot["qualified"]) if stop_reason is None else "dung"
    manifest = dict(task="T11b", review="2026-11-22 §6.2", started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), **git,
                    python=platform.python_version(), packages={name: version(name) for name in ("pandas", "pyarrow", "numpy")},
                    input_file=f"{paths['seckin_dir']}/{FILE_NAME}", input_sha256=file_sha256(source), input_bytes=source.stat().st_size, raw_accountid_dtype=raw_id_dtype,
                    episode_rules={k: list(v) if isinstance(v, tuple) else v for k, v in vars(rules).items()}, control_min=CONTROL_MIN, threshold=THRESHOLD)
    payload = dict(pilot=pilot, decision=decision, decision_rule=RULE_TEXT[decision], stop_reason=stop_reason, extra=extra, manifest=manifest)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    ids = {value for value in frame["accountid"].unique() if len(value) >= 8}
    if any(value in text for value in ids):
        raise SystemExit("JSON chứa accountid: không ghi.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"Đã ghi {out_path.relative_to(ROOT)}; decision = {decision}")
    if stop_reason == "không có ngày được phủ":
        raise SystemExit("Không có ngày được phủ trong tệp thử: dừng.")


if __name__ == "__main__":
    main()
