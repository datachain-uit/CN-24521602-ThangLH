# KLTN – Phát hiện hoạt động phối hợp theo thời gian (Iannucci et al., 2026)

Mã nguồn tái lập và mở rộng bài báo *Detecting Coordinated Activities Through Temporal, Multiplex, and Collaborative Analysis* (ICWSM 2026): tái lập Bảng 2 trên dữ liệu mô phỏng (T6, T6b), khung đánh giá theo mốc trên DS2 và các chiến dịch Seckin (T7, T9, T12), biến thể P6 + VC, bộ sinh DS2-v2 (T13) và kiểm tra tệp thử Russia_5 (T11b).

## 1. Cài đặt

Python 3.13 (các lần chạy đã ghi dùng 3.13.15 trên Windows 11). Chạy tại thư mục gốc dự án:

```powershell
python -m venv .venv-kltn
.\.venv-kltn\Scripts\python.exe -m pip install -r requirements-dev.txt
```

| Tệp | Nội dung |
|---|---|
| `requirements.txt` | Thư viện chạy mã dự án |
| `requirements-dev.txt` | `requirements.txt` + mã tham chiếu của tác giả, ghim commit `bae7b15` (cần cho test và các script T6b, T7) |
| `requirements-lock-windows.txt` | Môi trường đầy đủ đã dùng cho các lần chạy trong `reports/` (Windows) |

Muốn tái lập đúng phiên bản thư viện của các lần chạy đã ghi, cài `requirements-lock-windows.txt` thay cho `requirements-dev.txt`.

## 2. Mã tham chiếu của tác giả

Mã gốc: <https://github.com/letiziaia/time-aware-collaboration>, ghim tại commit `bae7b15b9e29e2c00dde585c5eaf210c10092730` (MIT License).

**Cách 1 – cài như gói Python** (đủ cho test và phần lớn script). Đã có trong `requirements-dev.txt`:

```text
time-aware-collaboration @ git+https://github.com/letiziaia/time-aware-collaboration@bae7b15b9e29e2c00dde585c5eaf210c10092730
```

**Cách 2 – clone vào `references/`** (cần thêm cho `scripts/diagnose_t6b.py`, script này so mã đang import với bản clone và ghi `git rev-parse HEAD` của nó; cũng cần nếu muốn chạy test của tác giả). Đường dẫn khai báo ở `reference_dir` trong `configs/paths.json`:

```powershell
git clone https://github.com/letiziaia/time-aware-collaboration references/time-aware-collaboration
git -C references/time-aware-collaboration checkout bae7b15b9e29e2c00dde585c5eaf210c10092730
```

Thư mục `references/` không được đưa vào repo.

## 3. Dữ liệu

Thư mục `data/` bị `.gitignore` chặn và không có trong repo. Các đường dẫn dữ liệu và thư mục kết quả khai báo ở `configs/paths.json`:

| Khóa | Đường dẫn | Dùng cho |
|---|---|---|
| `seckin_dir` | `data/Seckin` | T7, T9, T11b, T12, P6 + VC |
| `pote_dir` | `data/Coordinated_Reply_Attacks_dataset` | T4 |
| `german_dir` | `data/CooRTweet_dataset` | T4 |
| `partitions_dir` | `data/partitions` | Phân hoạch lưu bằng `--save-partitions` |

T6, T6b và T13 dùng dữ liệu mô phỏng sinh trong mã, không cần `data/`.

## 4. Chạy test

```powershell
$env:PYTHONHASHSEED = '0'
.\.venv-kltn\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

Test không cần `data/`. Các tệp `tests/test_t7.py` và `tests/test_delta_t.py` import `time_aware_collaboration`, nên phải cài `requirements-dev.txt` trước.

Chạy kèm test của tác giả (cần clone theo Cách 2 ở Mục 2):

```powershell
.\.venv-kltn\Scripts\python.exe -m pytest -p no:cacheprovider tests references/time-aware-collaboration/tests
```

## 5. Chạy lại các nhiệm vụ

Mọi lệnh chạy tại thư mục gốc dự án. Tham số mặc định (seed, chiến dịch, mốc, ngưỡng) đọc từ `configs/*.json`. Các script có `--output-dir` hoặc `--output-root` cho phép chạy thử ra thư mục khác mà không ghi đè `reports/`.

Khuyến nghị đặt trước khi chạy:

```powershell
$env:PYTHONHASHSEED = '0'
$env:TQDM_DISABLE = '1'
```

### 5.1. T6 – Tái lập Bảng 2 (Phụ lục B)

```powershell
.\.venv-kltn\Scripts\python.exe -m src.eval.reproduce_table2
```

Ra: `reports/t6_*.csv` (lưới β của ba mô phỏng, β tối ưu, Bảng 2 tái lập cho 7 cấu hình).

### 5.2. T6b – Phân phối 30 × 5 và chẩn đoán cấu hình 3

```powershell
.\.venv-kltn\Scripts\python.exe -m scripts.run_t6b_experiment            # thêm --pilot để chạy 1 data seed
.\.venv-kltn\Scripts\python.exe -m scripts.summarize_t6b
.\.venv-kltn\Scripts\python.exe -m scripts.diagnose_t6b --output-dir reports/t6b/diagnostics-repeat
```

Cấu hình: `configs/t6b.json`. Ra: `reports/t6b/full/`, thư mục chẩn đoán do `--output-dir` chỉ định (dùng thư mục mới cho mỗi lần). `diagnose_t6b` cần clone mã tham chiếu theo Cách 2.

### 5.3. T7 – Khung đánh giá theo mốc (DS2, Iran_5, Catalonia)

```powershell
.\.venv-kltn\Scripts\python.exe -m scripts.run_t7
```

Cấu hình: `configs/t7.json`. Ra: `reports/T7/t7_results.csv` (ghi nối, chạy tiếp theo khóa) và `reports/T7/run.json`. Tùy chọn: `--campaigns`, `--max-checkpoints`, `--seeds`, `--save-partitions`, `--output-dir`.

### 5.4. T9 – EDA và giao thức theo đợt cho 7 chiến dịch Seckin

Chạy notebook `notebooks/T9_EDA_Seckin.ipynb` từ kernel sạch. Cấu hình: `configs/t9.json`. Ra: `reports/T9_EDA_Seckin/`.

### 5.5. T12 – Giao thức đánh giá v3 trên kết quả T7

Cần `reports/T7/t7_results.csv` có sẵn (không chạy lại B0/B1).

```powershell
.\.venv-kltn\Scripts\python.exe -m scripts.t12_protocol_v3
.\.venv-kltn\Scripts\python.exe -m scripts.t12_tiebreak_checks
```

Cấu hình: `configs/t12.json`. Ra: `reports/T12/*.csv`, `run.json`, `run_tiebreak.json`.

### 5.6. P6 + VC – Biến thể P6 và VC (P5b)

Mỗi tiến trình chạy một nhóm chiến dịch và ghi shard riêng; lần chạy trong `reports/` dùng 3 tiến trình song song, mỗi tiến trình một luồng BLAS:

```powershell
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
.\.venv-kltn\Scripts\python.exe -m scripts.run_p6_vc --campaigns Catalonia --n-processes 3
.\.venv-kltn\Scripts\python.exe -m scripts.run_p6_vc --campaigns Thailand Spain Armenia --n-processes 3
.\.venv-kltn\Scripts\python.exe -m scripts.run_p6_vc --campaigns Iran_5 Ghana_Nigeria Bangladesh --n-processes 3
```

Ba lệnh chạy ở ba cửa sổ PowerShell riêng. Bị dừng giữa chừng thì chạy lại với `--resume`; `--check` chỉ đối chiếu khóa với shard. Sau khi cả ba xong:

```powershell
.\.venv-kltn\Scripts\python.exe -m scripts.analyze_p6_vc
```

Cấu hình: `configs/p6.json`, `configs/vc.json`. Ra: `reports/P6/`, `reports/VC/` (shard `results_<chiến dịch>.csv`, bảng tổng hợp, `run.json`). `run_p6_vc` từ chối chạy khi `src/`, `scripts/`, `configs/` hoặc `tests/` có thay đổi chưa commit hay tệp chưa track (`code_dirty`); `--allow-dirty` chỉ dùng được kèm `--output-root` khi chạy thử.

### 5.7. T13 – DS2-v2 (bộ sinh có ngụy trang)

```powershell
.\.venv-kltn\Scripts\python.exe -m scripts.run_t13
.\.venv-kltn\Scripts\python.exe -m scripts.summarize_t13
```

Cấu hình: `configs/t13.json`. Ra: `reports/T13/`. Chạy thử khói: `run_t13 --n-data-seeds 1 --output-dir <thư mục>` rồi `summarize_t13 --input-dir <thư mục> --n-data-seeds 1`. Chọn phần chạy bằng `--parts table2 modes grid`.

### 5.8. T11b – Kiểm tra độ phủ tệp thử Russia_5_part_1

```powershell
.\.venv-kltn\Scripts\python.exe -m scripts.t11b_pilot_russia5
```

Ra: `reports/T11/pilot_Russia_5_part_1.json` (chỉ thống kê tổng hợp, không có accountid hay nội dung bài đăng).

## 6. Cấu trúc mã nguồn

```text
configs/     cấu hình từng nhiệm vụ và paths.json
notebooks/   EDA (T4, T8, T9)
scripts/     điểm chạy của từng nhiệm vụ
src/
├── simulation/  synthetic.py (DS2), synthetic_v2.py (DS2-v2)
├── graph/       time_aware.py (Eq. 2)
├── community/   modularity.py, leiden.py
└── eval/        metrics.py, reproduce_table2.py, t7_checkpoints.py, seckin_protocol.py, protocol_v3.py, p6_variants.py
tests/       pytest
reports/     kết quả đã chạy
```

## 7. Lưu ý về mức độ tái lập

Bảng 2 là bản tái dựng dựa trên nội dung công bố, chưa được xác nhận là tái lập chính xác. Seed ngẫu nhiên, lịch sự kiện trong các đợt hoạt động và cách ánh xạ hành động vào layer không được bài báo quy định đầy đủ; các lựa chọn trong mã là giả định triển khai.

## 8. Tài liệu tham khảo

Iannucci, L., Muratore, E., Matakos, A., & Kivelä, M. (2026). *Detecting Coordinated Activities Through Temporal, Multiplex, and Collaborative Analysis.* ICWSM 2026. <https://arxiv.org/abs/2512.19677>
