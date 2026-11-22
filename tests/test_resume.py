"""Ghi liên tục và --resume của scripts/run_p6_vc.py: giết tiến trình giữa chừng (để lại dòng CSV cắt dở), chạy lại --resume, so với lần chạy liền mạch.

Tiến trình con chạy main() thật trên luồng tổng hợp (fixture T9) thay cho dữ liệu Seckin; đầu ra ở tmp_path, ngoài reports/.
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "Catalonia"


def drive(argv, kill_after=0):
    """Chạy trong tiến trình con: thay load_campaign bằng luồng tổng hợp; kill_after > 0 thì sau dòng kết quả thứ kill_after ghi nửa dòng rồi os._exit(1) (như bị giết khi đang ghi)."""
    from scripts import run_p6_vc
    from test_p6_variants import build_campaign
    run_p6_vc.load_campaign = lambda campaign, paths, t7: build_campaign()
    if kill_after:
        real = run_p6_vc.append_rows
        state = dict(count=0)

        def dying(path, rows, columns=run_p6_vc.COLUMNS):
            if columns is run_p6_vc.COLUMNS and state["count"] == kill_after:
                with open(path, "ab") as handle:
                    handle.write(b"P6,Catalonia,main,B1_chuan,2,1.0000,4")
                    handle.flush()
                    os.fsync(handle.fileno())
                os._exit(1)
            real(path, rows, columns)
            state["count"] += columns is run_p6_vc.COLUMNS
        run_p6_vc.append_rows = dying
    run_p6_vc.main(argv)


def run(out, *extra, kill_after=0):
    code = f"import sys; sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'tests')!r}]; import test_resume; test_resume.drive({['--campaigns', CAMPAIGN, '--output-root', str(out), '--allow-dirty', *extra]!r}, {kill_after})"
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=900)


def table(out):
    """Ghép shard kết quả (bỏ runtime_s) và bảng cụm control, sắp theo khóa."""
    from src.eval.p6_variants import KEY_COLUMNS, QUAL_KEY_COLUMNS
    results = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False) for path in sorted(Path(out).glob("*/results_*.csv"))], ignore_index=True)
    assert not results.duplicated(KEY_COLUMNS).any()
    qual = pd.read_csv(Path(out) / "P6" / f"control_clusters_{CAMPAIGN}.csv", dtype=str, keep_default_na=False)
    assert not qual.duplicated(QUAL_KEY_COLUMNS).any()
    return results.drop(columns="runtime_s").sort_values(KEY_COLUMNS, ignore_index=True), qual.sort_values(QUAL_KEY_COLUMNS, ignore_index=True)


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    out = tmp_path_factory.mktemp("straight")
    done = run(out)
    assert done.returncode == 0, done.stderr[-3000:]
    return out


def test_chay_lien_mach_du_khoa_va_run_json(straight):
    from src.eval.p6_variants import COLUMNS
    results, qual = table(straight)
    assert list(results.columns) == [c for c in COLUMNS if c != "runtime_s"] and len(qual) > 0
    for task in ("P6", "VC"):
        run_json = json.loads((straight / task / "run.json").read_text(encoding="utf-8"))
        assert run_json["campaigns"][CAMPAIGN]["complete"] and run_json["sessions"][-1]["status"] == "finished"
        assert run_json["config_hash"] and run_json["commit"] and "code_dirty" in run_json and run_json["sessions"][-1]["threads"].keys() == {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
        assert (straight / task / "run.log").read_text(encoding="utf-8").count("\tunit\t") == run_json["campaigns"][CAMPAIGN]["done_units"]


def test_giet_giua_chung_roi_resume_trung_lien_mach(straight, tmp_path):
    first = run(tmp_path, kill_after=5)
    assert first.returncode == 1
    shard_p6 = tmp_path / "P6" / f"results_{CAMPAIGN}.csv"
    shard_vc = tmp_path / "VC" / f"results_{CAMPAIGN}.csv"
    partial = shard_p6 if shard_p6.is_file() and not shard_p6.read_bytes().endswith(b"\n") else shard_vc
    assert not partial.read_bytes().endswith(b"\n")
    refused = run(tmp_path)
    assert refused.returncode != 0 and "--resume" in refused.stderr + refused.stdout
    second = run(tmp_path, "--resume", kill_after=9)
    assert second.returncode == 1
    for lock in (tmp_path / "VC" / "run.json.lock", tmp_path / "P6" / "run.log.lock"):
        lock.write_text(json.dumps(dict(pid=dead_pid(), create_time=1.0, host=platform.node())), encoding="utf-8")
    third = run(tmp_path, "--resume")
    assert third.returncode == 0, third.stderr[-3000:]
    assert "cắt dở" in third.stdout and third.stdout.count("gỡ khóa cũ") == 2
    assert "gỡ khóa cũ run.json.lock" in (tmp_path / "VC" / "run.log").read_text(encoding="utf-8")
    assert "gỡ khóa cũ run.log.lock" in (tmp_path / "P6" / "run.log").read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*/*.lock"))
    ours, ours_qual = table(tmp_path)
    ref, ref_qual = table(straight)
    pd.testing.assert_frame_equal(ours, ref)
    pd.testing.assert_frame_equal(ours_qual, ref_qual)
    run_json = json.loads((tmp_path / "VC" / "run.json").read_text(encoding="utf-8"))
    assert [item["status"] for item in run_json["sessions"]] == ["running", "running", "finished"]
    assert run_json["campaigns"][CAMPAIGN]["complete"]
    again = run(tmp_path, "--resume")
    assert again.returncode == 0
    assert table(tmp_path)[0].equals(ours)


def dead_pid():
    """PID của một tiến trình con đã kết thúc (như tiến trình chủ khóa khi máy tắt)."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_khoa_cua_tien_trinh_da_chet_duoc_go_va_ghi_log(tmp_path):
    from scripts import run_p6_vc
    run_path = tmp_path / "run.json"
    for content in (json.dumps(dict(pid=dead_pid(), create_time=1.0, host=platform.node())), json.dumps(dict(pid=os.getpid(), create_time=1.0, host=platform.node()))):
        Path(f"{run_path}.lock").write_text(content, encoding="utf-8")
        run_p6_vc.update_run(run_path, lambda payload: payload.update(n=payload.get("n", 0) + 1))
        assert not Path(f"{run_path}.lock").exists()
    assert json.loads(run_path.read_text(encoding="utf-8"))["n"] == 2
    run_p6_vc.log_line(tmp_path, "sau khi gỡ khóa")
    log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert log.count("gỡ khóa cũ run.json.lock") == 2 and log.index("gỡ khóa cũ") < log.index("sau khi gỡ khóa") and not run_p6_vc.STALE_EVENTS


def test_khoa_cua_tien_trinh_con_song_thi_cho(tmp_path):
    from scripts import run_p6_vc
    target = tmp_path / "run.json"
    with run_p6_vc.file_lock(target):
        owner = json.loads(Path(f"{target}.lock").read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid() and run_p6_vc.owner_alive(json.dumps(owner)) is True
        with pytest.raises(TimeoutError):
            with run_p6_vc.file_lock(target, timeout=0.3):
                pass
    assert not Path(f"{target}.lock").exists()
    Path(f"{target}.lock").write_text("", encoding="utf-8")
    with pytest.raises(TimeoutError):
        with run_p6_vc.file_lock(target, timeout=0.3):
            pass
    with run_p6_vc.file_lock(target, timeout=1.0, unreadable_after=0.0):
        pass
    run_p6_vc.STALE_EVENTS.clear()


def test_check_va_tu_choi_khi_head_khac(straight, tmp_path):
    check = run(straight, "--check")
    assert check.returncode == 0 and "'complete': True" in check.stdout
    out = tmp_path / "copy"
    run_first = run(out, kill_after=3)
    assert run_first.returncode == 1
    path = out / "P6" / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**payload, "commit": "0" * 40}), encoding="utf-8")
    refused = run(out, "--resume")
    assert refused.returncode != 0 and "TỪ CHỐI" in refused.stderr and "HEAD" in refused.stderr
