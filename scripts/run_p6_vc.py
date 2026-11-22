"""P6 + VC (Review-2026-11-21 §6.3–6.5): chạy theo chiến dịch, mỗi đơn vị ghi ngay một dòng vào shard reports/{P6,VC}/results_<campaign>.csv (fsync), có --resume theo khóa.

Mỗi tiến trình chỉ chạy các chiến dịch được giao và chỉ ghi shard của chúng; run.json ghi nguyên tử dưới khóa tệp vì nhiều tiến trình cùng cập nhật.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t7 import campaign_files, package_versions  # noqa: E402
from src.eval.p6_variants import COLUMNS, QUAL_COLUMNS, append_rows, ceil_k, io_marks, mark_key, method_rows, method_units, p6_rows, p6_units, qual_key, qualitative_rows, read_shard, unit_key, vc_marks, vc_stream  # noqa: E402
from src.eval.seckin_protocol import INPUT_COLUMNS, EpisodeRules, daily_coverage, harmonize_time, make_checkpoints, make_episodes  # noqa: E402
from src.eval.t7_checkpoints import format_tau, load_config  # noqa: E402

CONFIGS = {"p6": "configs/p6.json", "vc": "configs/vc.json"}
CODE_DIRS = ["src", "scripts", "configs", "tests"]
THREAD_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
P2_FILE = "p2_account_day_activity.csv"
NOTES = {
    "main": "T7 chạy ở commit khác với code_dirty = True nên không dùng lại; P6 chạy lại biến thể main trong cùng lần chạy.",
    "b1": "B1-chuẩn = keep_zero True, B1-gốc = keep_zero False; ghép cặp một lần mỗi (mốc, biến thể dữ liệu), β chọn một lần mỗi (mốc, biến thể dữ liệu, biến thể B1, lớp) với community_seeds[0] của configs/t7.json; seed chỉ cho Leiden (và RNG xáo trộn S1 của shuffle_account_day).",
    "u2": "U2 (t ≤ tau) kiểm ở observe, sau prepare_variant (mọi biến thể, kể cả cap/shuffle), trong checkpoint_pairs/select_beta_brent, trivial_rows và trước P5.",
    "vc": "VC: cap_volume lên luồng sau make_episodes; assert ngày được phủ và đợt (start, end, qualified) không đổi; mốc q tính lại trên luồng đã cap (tau_vc, eligible_vc); tau_goc, eligible_goc lấy từ giao thức gốc. Không assert eligible theo mốc (đã quyết định: một số mốc đổi trạng thái).",
    "goc": "protocol = GOC: P5 và TRIV trên luồng gốc ở 4 mốc q và mốc theo ngày, chạy lại cùng commit, làm tham chiếu trước/sau VC và giao thức gốc. B1-chuẩn GOC lấy từ P6 main (chỉ q = 25%, 100%); B1-gốc không có so sánh trước/sau VC.",
    "day_marks": "Mốc theo ngày (day_values của configs/vc.json) chỉ chạy P5 và TRIV (không B1), ở cả GOC và VC, để chọn TRIV_LOCO đúng như loco_choice của T12 (tag t12-v1) trên mọi mốc io_pct và theo ngày eligible.",
    "lock": "Tệp khóa ghi PID + create_time; khóa của tiến trình không còn sống được gỡ và ghi vào run.log.",
    "ineligible": "Mốc không đạt: mỗi đơn vị một dòng với chỉ số trống; B1 mốc không đạt ghi seed trống.",
    "code_dirty": "git status --porcelain -- src scripts configs tests (gồm cả untracked trong 4 thư mục này, như T7); reports/ không bị xét.",
    "runtime_s": "B1: (chuẩn bị biến thể + ghép cặp) chia đều cho các đơn vị của lần dựng + dựng đồ thị / số seed + Leiden của seed. TRIV/P5: thời gian chia đều cho các dòng.",
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_all():
    """paths.json, t7.json (qua load_config), p6.json, vc.json."""
    paths = json.loads((ROOT / "configs" / "paths.json").read_text(encoding="utf-8"))
    cfgs = {name: json.loads((ROOT / path).read_text(encoding="utf-8")) for name, path in CONFIGS.items()}
    return paths, load_config(), cfgs["p6"], cfgs["vc"]


def config_hash(t7, p6, vc, p2_path):
    """sha256 của JSON chuẩn hóa {p6, vc, t7, sha256(p2_account_day_activity.csv)}."""
    payload = dict(p6=p6, vc=vc, t7=t7, p2_sha256=hashlib.sha256(Path(p2_path).read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def check_k(p6, vc, p2_path, control_min):
    """cap_k_p6 = ceil(median), K_c = ceil(P90) của IO trên covered_c{control_min} trong p2_account_day_activity.csv; lệch hoặc thiếu thì dừng."""
    table = pd.read_csv(p2_path)
    table = table.loc[table["scope"].eq(f"covered_c{control_min}") & table["label"].eq("IO")].set_index("campaign")
    for name, campaigns, values, column in (("cap_k_p6", p6["campaigns"], p6["cap_k_p6"], "median"), ("k_c", vc["p5_campaigns"], vc["k_c"], "p90")):
        for campaign in campaigns:
            if campaign not in values or campaign not in table.index:
                raise ValueError(f"Thiếu {name} hoặc dòng p2 cho {campaign}.")
            expected = ceil_k(table.loc[campaign, column])
            if int(values[campaign]) != expected:
                raise ValueError(f"{name}[{campaign}] = {values[campaign]} khác ceil({column}) = {expected} của T9. DỪNG.")


def git_state():
    """HEAD, tag trỏ vào HEAD, code_dirty như T7 (git status --porcelain -- src scripts configs tests)."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return dict(commit=run("rev-parse", "HEAD") or None, tags=run("tag", "--points-at", "HEAD").split(), code_dirty=bool(run("status", "--porcelain", "--", *CODE_DIRS)))


STALE_EVENTS = []


def process_identity():
    """PID, thời điểm tạo tiến trình (psutil) và tên máy, ghi vào tệp khóa để nhận ra khóa của tiến trình đã chết (kể cả PID bị dùng lại sau khi khởi động lại máy)."""
    try:
        import psutil
        created = psutil.Process().create_time()
    except ImportError:
        created = None
    return dict(pid=os.getpid(), create_time=created, host=platform.node())


def owner_alive(text):
    """True/False theo PID + create_time trong tệp khóa (cùng máy); None nếu không đọc được nội dung, khác máy hoặc thiếu psutil."""
    try:
        owner = json.loads(text)
        import psutil
    except (ValueError, ImportError):
        return None
    if owner.get("host") not in (None, platform.node()):
        return None
    if not psutil.pid_exists(owner["pid"]):
        return False
    try:
        created = psutil.Process(owner["pid"]).create_time()
    except psutil.NoSuchProcess:
        return False
    return owner.get("create_time") is None or abs(created - owner["create_time"]) < 1e-3


@contextmanager
def file_lock(target, timeout=300.0, unreadable_after=60.0):
    """Khóa liên tiến trình bằng tạo tệp độc quyền <target>.lock chứa PID + create_time của chủ khóa.

    Khóa của tiến trình không còn sống (máy tắt, bị giết) bị gỡ và ghi vào STALE_EVENTS (thư mục, thời điểm, thông báo); log_line đưa vào run.log của cùng thư mục. Tệp khóa không đọc được (chết ngay sau khi tạo) bị gỡ khi cũ hơn unreadable_after giây.
    """
    lock = Path(f"{target}.lock")
    start = time.time()
    text = ""
    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"Không lấy được khóa {lock} (chủ khóa: {text.strip() or 'không đọc được'}).")
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except PermissionError:
            time.sleep(0.02)  # Windows: tệp khóa cũ đang chờ xóa (delete pending) hoặc đang bị tiến trình khác mở
            continue
        except FileExistsError:
            try:
                text, age = lock.read_text(encoding="utf-8"), time.time() - lock.stat().st_mtime
            except (FileNotFoundError, PermissionError):
                time.sleep(0.02)
                continue
            alive = owner_alive(text)
            if alive is False or (alive is None and age > unreadable_after):
                try:
                    lock.unlink()
                except (FileNotFoundError, PermissionError):
                    time.sleep(0.02)
                    continue
                message = f"CẢNH BÁO: gỡ khóa cũ {lock.name} của tiến trình không còn sống ({text.strip() or 'tệp rỗng'}, {age:.0f}s)."
                STALE_EVENTS.append((lock.parent.resolve(), now(), message))
                print(message, flush=True)
                continue
            time.sleep(0.05)
            continue
        try:
            os.write(handle, json.dumps(process_identity()).encode("utf-8"))
            os.fsync(handle)
        finally:
            os.close(handle)
        break
    try:
        yield
    finally:
        retry_windows(lambda: lock.unlink(missing_ok=True), f"gỡ khóa {lock}")


def retry_windows(action, what, timeout=30.0):
    """Thử lại khi Windows báo tệp đang bị tiến trình khác mở (WinError 32/5, PermissionError); hết thời gian thì báo lỗi."""
    start = time.time()
    while True:
        try:
            return action()
        except PermissionError:
            if time.time() - start > timeout:
                raise
            time.sleep(0.01)


def write_json_atomic(path, payload):
    temp = Path(f"{path}.tmp{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    retry_windows(lambda: os.replace(temp, path), f"thay {path}")


def update_run(path, change):
    """Đọc–sửa–ghi run.json dưới khóa; change(payload) sửa tại chỗ."""
    with file_lock(path):
        payload = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}
        change(payload)
        write_json_atomic(path, payload)
        return payload


def log_line(directory, text):
    """Ghi một dòng vào run.log (append, dưới khóa, fsync); trước đó ghi các sự kiện gỡ khóa cũ thuộc cùng thư mục."""
    path = Path(directory) / "run.log"
    with file_lock(path):
        with path.open("a", encoding="utf-8") as handle:
            for event in [event for event in STALE_EVENTS if event[0] == Path(directory).resolve()]:
                STALE_EVENTS.remove(event)
                handle.write(f"{event[1]}\t{event[2]}\n")
            handle.write(f"{now()}\t{text}\n")
            handle.flush()
            os.fsync(handle.fileno())


def peak_memory_mb():
    """Bộ nhớ đỉnh của tiến trình (Windows: peak working set; nơi khác: ru_maxrss)."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            return round(peak / 2**20, 1)
    except ImportError:
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except ImportError:
        return None


def guard(run_path, state, digest, args):
    """Từ chối khi run.json có sẵn khác HEAD/config_hash, hoặc code bẩn (trừ --allow-dirty ở thư mục tạm)."""
    if state["code_dirty"] and not args.allow_dirty:
        raise SystemExit("TỪ CHỐI: code_dirty = True (src/ scripts/ configs/ tests/ có thay đổi hoặc file chưa track). Commit + tag trước khi chạy.")
    if not Path(run_path).is_file():
        return
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    problems = []
    if run.get("commit") != state["commit"]:
        problems.append(f"HEAD {state['commit']} ≠ commit trong run.json {run.get('commit')}")
    if run.get("config_hash") != digest:
        problems.append("config_hash khác run.json")
    if run.get("code_dirty") and not args.allow_dirty:
        problems.append("run.json ghi code_dirty = True")
    if problems:
        raise SystemExit(f"TỪ CHỐI chạy tiếp ({run_path}): " + "; ".join(problems) + ".")


def base_run(task, state, digest, cfgs, args):
    return dict(task=task, review="Review-2026-11-21 §6.3–6.5; Review-2026-11-22", created_utc=now(), **state, config_hash=digest, configs=cfgs, packages=package_versions(), python=sys.version.split()[0], platform=platform.platform(), cpu_count=os.cpu_count(), notes=NOTES, sessions=[], campaigns={})


def load_campaign(campaign, paths, t7):
    """P1–P4 bằng hàm T9 (như load_ds1 của T7) và đối chiếu mốc với p4_checkpoints.csv của T9; trả stream, daily, episodes, checkpoints."""
    rules = EpisodeRules()
    frame = harmonize_time(pd.concat([pd.read_parquet(path, columns=INPUT_COLUMNS) for path in campaign_files(ROOT / paths["seckin_dir"], campaign)], ignore_index=True))
    daily = daily_coverage(frame, rules)
    stream, episodes = make_episodes(frame, daily, rules, t7["control_min"])
    del frame
    checkpoints = make_checkpoints(stream, episodes, rules)
    reference = pd.read_csv(ROOT / paths["t9_output_dir"] / "p4_checkpoints.csv")
    reference = reference.loc[reference["campaign"].eq(campaign)].reset_index(drop=True)
    ours = checkpoints.reset_index(drop=True)
    same = len(ours) == len(reference) and ours[["episode", "n_io", "n_ctrl", "eligible"]].astype(str).equals(reference[["episode", "n_io", "n_ctrl", "eligible"]].astype(str)) and ours["tau"].map(format_tau).equals(reference["tau"].map(format_tau))
    if not same:
        raise ValueError(f"Mốc của {campaign} không khớp p4_checkpoints.csv của T9.")
    return stream, daily, episodes, checkpoints, rules


def select(marks, args):
    """Lọc mốc cho chạy thử (--episodes, --q); mặc định giữ nguyên."""
    if args.episodes:
        marks = marks.loc[marks["episode"].isin(args.episodes)]
    if args.q:
        marks = marks.loc[np.isclose(marks["q"].to_numpy(dtype=float)[:, None], np.asarray(args.q)[None, :]).any(axis=1)]
    return marks.reset_index(drop=True)


def ordered(marks, q_main):
    """Mốc q_main chạy trước, rồi các q còn lại tăng dần; giữ thứ tự đợt."""
    first = np.isclose(marks["q"].astype(float), q_main)
    return pd.concat([marks.loc[first], marks.loc[~first].sort_values(["q", "episode"], kind="stable")], ignore_index=True)


def plan_jobs(campaign, data, t7, p6, vc, args):
    """Danh sách việc (protocol, thư mục, mốc, hàm sinh dòng, hàm liệt kê khóa) của một chiến dịch, mốc q_main trước."""
    stream, daily, episodes, checkpoints, rules = data
    seeds_p6 = args.seeds or p6["seeds"]
    seeds_vc = args.seeds or vc["seeds"]
    jobs = []
    if campaign in p6["campaigns"]:
        variants = [*p6["variants"], *(["reset_state"] if campaign in p6["reset_state_campaigns"] else [])]
        cap_k = int(p6["cap_k_p6"][campaign])
        qual = p6["qualitative"]
        for mark in ordered(select(io_marks(checkpoints, p6["q_values"]), args), p6["interpretation"]["q_main"]).to_dict("records"):
            keys = [mark_key("P6", campaign, variant, p6["method"], mark, seed, args.config_hash) for variant, seed in p6_units(mark, variants, seeds_p6)]
            wants_qual = campaign in qual["campaigns"] and np.isclose(mark["q"], qual["q"]) and qual["seed"] in seeds_p6
            jobs.append(dict(protocol="P6", mark=mark, keys=keys, run=lambda done, mark=mark, variants=variants, cap_k=cap_k, qualitative=None: p6_rows(stream, mark, campaign, variants, seeds_p6, cap_k, t7, args.config_hash, done, qualitative), qual=wants_qual))
    if campaign in vc["p5_campaigns"]:
        goc = io_marks(checkpoints, vc["q_values"], vc["day_values"])
        capped = vc_stream(stream, daily, episodes, vc["k_c"][campaign], rules, t7["control_min"])
        vcm = vc_marks(capped, episodes, rules, vc["q_values"], goc, vc["day_values"])
        has_methods = campaign in vc["method_campaigns"]
        for protocol, variant, source, marks, b1 in (("GOC", "main", stream, goc, []), ("VC", "vc", capped, vcm, vc["b1_methods"] if has_methods else [])):
            for mark in ordered(select(marks, args), vc["comparison"]["q_main"]).to_dict("records"):
                b1_mark = b1 if mark["checkpoint_type"] == "io_pct" else []
                keys = [mark_key(protocol, campaign, variant, method, mark, seed, args.config_hash) for method, seed in method_units(mark, b1_mark, seeds_vc, has_methods, True)]
                jobs.append(dict(protocol=protocol, mark=mark, keys=keys, run=lambda done, protocol=protocol, variant=variant, source=source, mark=mark, b1=b1_mark: method_rows(protocol, variant, source, mark, campaign, b1, seeds_vc, has_methods, True, t7, args.config_hash, done), qual=False))
    order = {"GOC": 0, "VC": 1, "P6": 2}
    q_main = p6["interpretation"]["q_main"]
    return sorted(jobs, key=lambda job: (not np.isclose(job["mark"]["q"], q_main), order[job["protocol"]]))


def run_campaign(campaign, dirs, cfgs, paths, args, session):
    """Chạy một chiến dịch; trả (số dòng mới, số khóa dự kiến, số khóa đã có) theo thư mục."""
    t7, p6, vc = cfgs["t7"], cfgs["p6"], cfgs["vc"]
    clock = perf_counter()
    data = load_campaign(campaign, paths, t7)
    jobs = plan_jobs(campaign, data, t7, p6, vc, args)
    shard = {protocol: dirs["P6" if protocol == "P6" else "VC"] / f"results_{campaign}.csv" for protocol in ("P6", "VC", "GOC")}
    qual_path = dirs["P6"] / f"control_clusters_{campaign}.csv"
    warn = lambda directory: (lambda text: (print(text, flush=True), None if args.check else log_line(directory, f"{campaign}\t{text}")))
    done = {path: read_shard(path, warn=warn(path.parent), repair=not args.check) for path in set(shard.values())}
    qual_done = read_shard(qual_path, QUAL_COLUMNS, qual_key, warn(dirs["P6"]), repair=not args.check)
    if not args.resume and not args.check and any(done.values()):
        raise SystemExit(f"TỪ CHỐI: shard của {campaign} đã có dòng; dùng --resume để chạy tiếp.")
    expected = {path: set() for path in done}
    written = 0
    for job in jobs:
        path = shard[job["protocol"]]
        expected[path].update(unit_key(key) for key in job["keys"])
        if args.check:
            continue
        qualitative = None
        if job["qual"]:
            mark = job["mark"]

            def callback(rows, mark=mark):
                fresh = [row for row in qualitative_rows(rows, mark, campaign, p6["qualitative"]["seed"], args.config_hash) if qual_key(row) not in qual_done]
                if fresh:
                    append_rows(qual_path, fresh, QUAL_COLUMNS)
                    qual_done.update(qual_key(row) for row in fresh)
            qualitative = dict(seed=p6["qualitative"]["seed"], top=p6["qualitative"]["top"], callback=callback)
        rows = job["run"](done[path], qualitative=qualitative) if job["protocol"] == "P6" else job["run"](done[path])
        for row in rows:
            key = unit_key(row)
            if key in done[path]:
                raise AssertionError(f"Khóa trùng: {key}")
            append_rows(path, [row])
            done[path].add(key)
            written += 1
            log_line(path.parent, f"{campaign}\tunit\t{'|'.join(key[:-1])}\truntime_s={row['runtime_s'] if not pd.isna(row['runtime_s']) else ''}")
        print(f"[{campaign}] {job['protocol']} episode={job['mark']['episode']} {job['mark']['checkpoint_type']}={job['mark']['checkpoint_value']}: {sum(len(done[p]) for p in done)} dòng | {perf_counter() - clock:.0f}s", flush=True)
    summary = {}
    for path, keys in expected.items():
        if not keys and not done[path]:
            continue
        missing, extra = keys - done[path], done[path] - keys
        summary[path] = dict(expected_units=len(keys), done_units=len(done[path] & keys), missing_units=len(missing), extra_units=len(extra), complete=not missing and not extra)
        if extra:
            raise AssertionError(f"{path.name} có {len(extra)} khóa ngoài kế hoạch.")
    return written, summary, perf_counter() - clock


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaigns", nargs="+", required=True, help="Chiến dịch tiến trình này chạy (tối đa 1 tiến trình mỗi chiến dịch).")
    parser.add_argument("--resume", action="store_true", help="Chạy tiếp theo khóa khi shard đã có dòng.")
    parser.add_argument("--check", action="store_true", help="Chỉ đối chiếu khóa dự kiến với shard, không chạy mô hình.")
    parser.add_argument("--n-processes", type=int, default=1, help="Số tiến trình song song của cả lần chạy (ghi vào run.json).")
    parser.add_argument("--output-root", default=None, help="Chạy thử: thư mục gốc chứa P6/ và VC/ (không phải reports/).")
    parser.add_argument("--allow-dirty", action="store_true", help="Chỉ cho chạy thử với --output-root: cho phép code_dirty = True.")
    parser.add_argument("--episodes", nargs="+", type=int, default=None, help="Chạy thử: chỉ các đợt này.")
    parser.add_argument("--q", nargs="+", type=float, default=None, help="Chạy thử: chỉ các mốc q này.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Chạy thử: seed thay cho config.")
    args = parser.parse_args(argv)
    if (args.allow_dirty or args.episodes or args.q or args.seeds) and not args.output_root:
        raise SystemExit("--allow-dirty/--episodes/--q/--seeds chỉ dùng với --output-root (chạy thử ngoài reports/).")
    return args


def main(argv=None):
    for console in (sys.stdout, sys.stderr):
        console.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    paths, t7, p6, vc = load_all()
    p2_path = ROOT / paths["t9_output_dir"] / P2_FILE
    check_k(p6, vc, p2_path, t7["control_min"])
    args.config_hash = config_hash(t7, p6, vc, p2_path)
    cfgs = dict(t7=t7, p6=p6, vc=vc)
    if args.output_root:
        root = Path(args.output_root).resolve()
        if root == ROOT or ROOT / "reports" in [root, *root.parents]:
            raise SystemExit("--output-root phải nằm ngoài reports/ của repo.")
        dirs = {"P6": root / "P6", "VC": root / "VC"}
    else:
        dirs = {"P6": ROOT / p6["output_dir"], "VC": ROOT / vc["output_dir"]}
    unknown = [c for c in args.campaigns if c not in p6["campaigns"] and c not in vc["p5_campaigns"]]
    if unknown:
        raise SystemExit(f"Chiến dịch không có trong config: {unknown}.")
    if not args.resume and not args.check:
        existing = [path.name for campaign in args.campaigns for directory in dirs.values() for path in [directory / f"results_{campaign}.csv"] if path.is_file() and path.stat().st_size]
        if existing:
            raise SystemExit(f"TỪ CHỐI: đã có shard {existing}; dùng --resume để chạy tiếp.")
    state = git_state()
    session = dict(session_id=uuid.uuid4().hex[:8], pid=os.getpid(), campaigns=args.campaigns, started_utc=now(), ended_utc=None, status="running", units_written=0, n_processes=args.n_processes, threads={name: os.environ.get(name) for name in THREAD_VARS}, args={k: v for k, v in vars(args).items() if k != "config_hash"}, peak_memory_mb=None)
    for task, directory in dirs.items():
        if args.check:
            guard(directory / "run.json", state, args.config_hash, args)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        run_path = directory / "run.json"
        with file_lock(run_path):
            guard(run_path, state, args.config_hash, args)
            payload = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else base_run(task, state, args.config_hash, cfgs, args)
            payload["sessions"].append(dict(session))
            write_json_atomic(run_path, payload)
        log_line(directory, f"session {session['session_id']} start pid={os.getpid()} campaigns={args.campaigns} commit={state['commit']} dirty={state['code_dirty']}")
    print(f"commit {state['commit']} tags {state['tags']} code_dirty={state['code_dirty']} config_hash={args.config_hash[:12]}", flush=True)
    status = "finished"
    total = 0
    try:
        for campaign in args.campaigns:
            written, summary, seconds = run_campaign(campaign, dirs, cfgs, paths, args, session)
            total += written
            for path, info in summary.items():
                task = "P6" if path.parent == dirs["P6"] else "VC"

                def change(payload, campaign=campaign, info=info, seconds=seconds, task=task):
                    payload["campaigns"][campaign] = dict(**info, updated_utc=now())
                    for item in payload["sessions"]:
                        if item["session_id"] == session["session_id"]:
                            item.update(units_written=total, peak_memory_mb=peak_memory_mb(), runtime_s_by_campaign={**item.get("runtime_s_by_campaign", {}), campaign: round(seconds, 1)})
                if not args.check:
                    update_run(dirs[task] / "run.json", change)
                print(f"[{campaign}] {task}: {info}", flush=True)
            if not args.check:
                print(f"[{campaign}] xong: {written} dòng mới, {seconds:.0f}s, bộ nhớ đỉnh {peak_memory_mb()} MB.", flush=True)
    except BaseException as error:
        status = f"error: {type(error).__name__}: {error}"
        raise
    finally:
        if not args.check:
            for directory in dirs.values():
                def close(payload):
                    for item in payload["sessions"]:
                        if item["session_id"] == session["session_id"]:
                            item.update(ended_utc=now(), status=status, units_written=total, peak_memory_mb=peak_memory_mb())
                update_run(directory / "run.json", close)
                log_line(directory, f"session {session['session_id']} end status={status} units_written={total} peak_memory_mb={peak_memory_mb()}")


if __name__ == "__main__":
    main()
