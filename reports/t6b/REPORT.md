# T6b — Chẩn đoán tái lập Bảng 2

Ngày 18/11/2026 · Review 03, mục 6.4. [Yêu cầu của GVHD](../../reviews/Review-2026-11-15.md#64-chẩn-đoán-t6-cấu-hình-3-t6b-tối-đa-2-ngày)

**T6b đủ điều kiện Done theo mục 6.4.** U1-gốc đạt ở cấp cạnh và U4 đạt theo phân phối cho cả hai pipeline. Đối chiếu độc lập với mã T10 vô hướng và β đã lưu cho phép tiếp tục sử dụng 2.100 dòng kết quả hiện có.

## Số cặp Δt = 0

Một cặp trong bảng là một kết quả ghép giữa hai tài khoản trên cùng `(action_type, content_id)`. Với `u < v` theo ID chuỗi, `u → v` tìm thời điểm sau hẳn, còn `v → u` cho phép trùng thời điểm. Chẩn đoán giữ các cặp 0 để đếm. Cột trước dùng phút liên tục của bộ sinh. Cột sau dùng `np.rint(timestamp * 60) / 60` của runner lịch sử, tức làm tròn đến giây gần nhất với nửa giây làm tròn về số chẵn. [CSV theo mô phỏng và từng seed](diagnostics-2026-11-18/delta_t_zero_by_seed.csv)

| Mô phỏng | Tổng cặp trước | Tổng cặp sau | Δt = 0 trước | Δt = 0 sau | Seed có cặp 0 sau làm tròn |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 602.464 | 602.464 | 0 | 3 | 20261114, 20261115, 20261132 |
| 2 | 603.160 | 603.158 | 0 | 3 | 20261115, 20261127, 20261133 |
| 3 | 605.173 | 605.173 | 0 | 2 | 20261115, 20261134 |
| **Tổng** | **1.810.797** | **1.810.795** | **0** | **8** | **8 tổ hợp mô phỏng–seed** |

Làm tròn có thể gộp hai thời điểm khác nhau vào cùng một giây. Khi đó, chiều `v → u` có thể chọn một đích bằng thời điểm nguồn. Với `keep_zero=False`, kết quả 0 bị bỏ sau khi ghép, không tìm lại đích dương. Chiều `u → v` bỏ qua đích bằng thời điểm nguồn và có thể ghép với một đích muộn hơn. Trong dữ liệu này, làm tròn tạo tám cặp 0 và làm số cặp dương giảm ròng mười. Tất cả điểm lịch sử đã dùng timestamp làm tròn.

## U1-gốc cấp cạnh

SV và `get_time_damped_edge_list` nhận cùng dữ liệu đã làm tròn và cùng β. Mã tham chiếu nhận lịch sử đến giây và danh sách tài khoản đã sắp theo ID để áp dụng đúng hai chiều canonical. Cả hai tính Δt theo phút rồi dùng `exp(−β × Δt / 1440) / (n_k − 1)`, trong đó `n_k` là số tài khoản phân biệt có cùng hành động và nội dung. Cả hai bỏ Δt = 0 và chỉ giữ cạnh có **tổng** trọng số `> 1e-9`. Với mỗi cặp mô phỏng–seed, phép kiểm dùng β bằng 0, β đã lưu của SV, β đã lưu của tham chiếu và β bằng 10. Sai số tính trên hợp hai tập cạnh với trọng số 0 cho cạnh vắng mặt. Hai tập cạnh còn được đối chiếu trực tiếp. [CSV U1-gốc](diagnostics-2026-11-18/u1_original_edge_weights.csv)

**Đạt.** Cả 360 phép so sánh cho tập cạnh trùng nhau. Sai số tuyệt đối lớn nhất là **8,881784197001252 × 10⁻¹⁶**, dưới ngưỡng nghiêm ngặt `1e-9`. Sai số lớn nhất xuất hiện ở mô phỏng 2, seed `20261133`, β = 9,85.

## Kiểm chứng kho kết quả

- `results_all.csv` có đủ tích Descartes 30 data seed × 5 community seed × 7 cấu hình × 2 pipeline. Các khóa duy nhất và khớp 30 tệp `results_<seed>.csv`. `betas_all.csv` và 30 tệp β theo seed khớp 180 tổ hợp. Đã kiểm tra β hiện diện đúng theo lớp, nằm trong miền, khớp lưới và cờ biên của SV. Seed tối ưu của tham chiếu bằng `data_seed × 10 + simulation`. Các chỉ số hữu hạn và nằm trong miền đến sai số số học `1e-12`.
- Runner lịch sử chọn β cho SV trên lưới `[0, 10]` với 1.001 điểm, bước 0,01, lấy trung bình Q của năm community seed rồi chọn argmax. SV dùng Leiden Modularity và trọng số lớp bằng nhau. Tham chiếu dùng `calculate_optimal_beta` của tác giả với bounded Brent và Louvain, sau đó dùng greedy modularity cho đơn lớp, Leiden Surprise với trọng số lớp bằng số cạnh cho đa lớp. SV có 22/30, 20/30 và 13/30 lần chạm biên trên ở mô phỏng 1, 2 và 3, tương ứng báo cáo `β ≥ 10`. β trung vị của tham chiếu lần lượt là 9,4164, 9,2747 và 9,2641. [Cấu hình](../../configs/t6b.json) · [β lịch sử](full/beta_summary.csv)
- Kiểm tra tác động của T10 dùng mã vô hướng ở `0db9c18812928b93507a122a0ee6275737cd3a03` làm mốc. Cả mốc và working tree đều áp dụng hai chiều strict/non-strict của T10. Bảng đầu vào tính trọng số `(u, v, Δt theo ngày, 1/(n_k−1))` trùng chính xác, kể cả thứ tự dòng, ở cả 90 tổ hợp mô phỏng–seed. Số cạnh dựng lại khớp 180/180 đồ thị tại β đã lưu. Q trung bình của SV trên năm community seed khớp 90/90 giá trị, với sai số tuyệt đối lớn nhất bằng 0. T10 vector hóa bước ghép nhưng cho cùng đầu vào tính trọng số trên mọi β của các đồ thị đã xét. Vì vậy tiếp tục dùng kết quả lưu trữ và chỉ kiểm chứng tại β đã lưu, không chạy lại quá trình tìm β và đánh giá toàn bộ cấu hình. [CSV đối chiếu đồ thị lưu trữ](diagnostics-2026-11-18/archive_graph_validation.csv)
- Khi dựng đồ thị trong kho lịch sử, pipeline tham chiếu giữ toàn bộ ứng viên từ `get_time_damped_edge_list`. Trong 90 đồ thị tham chiếu, có tổng cộng 4.071 lượt cạnh với `w ≤ 1e-9`. SV dùng `w > 1e-9`. Điều kiện lọc chung thuộc phép kiểm U1-gốc có kiểm soát. Hai pipeline lịch sử cũng có thể chọn β và phân hoạch khác nhau.

## Phân phối chỉ số và U4

Mỗi ô chỉ số ghi `median [P10–P90]`, với phân vị nội suy tuyến tính. Ba cấu hình đơn lớp của tham chiếu có năm dòng cùng phân hoạch cho mỗi data seed, nên mỗi cấu hình dùng 30 kết quả greedy. Các nhóm có stochastic Leiden giữ đủ 150 lần đánh giá trên 30 data seed. Các lần dùng chung dữ liệu là quan sát lồng theo data seed. P10–P90 mô tả phân phối điểm, không phải khoảng tin cậy. Chỉ cắt phần sai lệch số học dưới `1e-12` ở biên `[0, 1]` trước khi tính phân vị. [CSV phân phối](diagnostics-2026-11-18/summary_statistics.csv) · [CSV U4](diagnostics-2026-11-18/u4_distribution.csv)

| Pipeline | Cấu hình | Số kết quả | F1* | WP | Homogeneity |
| --- | --- | ---: | --- | --- | --- |
| SV | 1 | 150 | 0,923 [0,750–1,000] | 0,857 [0,600–1,000] | 0,839 [0,574–1,000] |
| SV | 2 | 150 | 0,923 [0,702–1,000] | 0,857 [0,541–1,000] | 0,839 [0,570–1,000] |
| SV | 3 | 150 | 0,500 [0,286–0,800] | 0,400 [0,186–0,718] | 0,371 [0,122–0,687] |
| SV | 1+2 | 150 | 1,000 [0,800–1,000] | 1,000 [0,667–1,000] | 1,000 [0,678–1,000] |
| SV | 1+3 | 150 | 1,000 [0,750–1,000] | 1,000 [0,600–1,000] | 1,000 [0,622–1,000] |
| SV | 2+3 | 150 | 1,000 [0,800–1,000] | 1,000 [0,667–1,000] | 1,000 [0,678–1,000] |
| SV | 1+2+3 | 150 | 1,000 [0,923–1,000] | 1,000 [0,857–1,000] | 1,000 [0,839–1,000] |
| Tham chiếu | 1 | 30 | 0,890 [0,706–1,000] | 0,804 [0,545–1,000] | 0,793 [0,574–1,000] |
| Tham chiếu | 2 | 30 | 0,857 [0,667–0,923] | 0,750 [0,500–0,857] | 0,747 [0,533–0,839] |
| Tham chiếu | 3 | 30 | 0,552 [0,305–0,811] | 0,492 [0,194–0,858] | 0,479 [0,119–0,840] |
| Tham chiếu | 1+2 | 150 | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] |
| Tham chiếu | 1+3 | 150 | 1,000 [0,800–1,000] | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] |
| Tham chiếu | 2+3 | 150 | 1,000 [0,800–1,000] | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] |
| Tham chiếu | 1+2+3 | 150 | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] |

**U4 đạt cho cấu hình 3 ở cả hai pipeline.** F1* công bố là 0,67, nằm trong [0,286–0,800] của SV và [0,305–0,811] của tham chiếu. Tỷ lệ đạt `F1* ≥ 0,67` lần lượt là **29/150 (19,3%)** và **9/30 (30,0%)**. Tỷ lệ dùng ngưỡng 0,67 trên điểm chưa làm tròn.

## Giả định tái dựng Phụ lục B

Phụ lục B mô tả 6 tài khoản phối hợp với tổng 15–20 sự kiện, 40 tài khoản thường với 1.000 sự kiện, thời gian nền phân bố đều trên 7 ngày, cùng danh mục 20 nội dung thuộc 3 loại. Bộ sinh hiện tại lấy ngẫu nhiên tài khoản thường với hoàn lại, dùng danh mục hashtag/retweet/mention theo tỷ lệ 7/7/6 và chuẩn hóa trọng số từ hỗn hợp chuẩn theo Phụ lục B sau khi đặt sàn dương `0.001`. Các dòng phối hợp chọn loại hành động đều, chọn hai nội dung khác nhau cho mỗi loại, rồi lấy theo xác suất `0.9/0.1` như mô tả của tác giả. Từng mô phỏng là một lớp trong Bảng 2. Ba loại hành động là khóa ghép trong lớp. Các mô phỏng dùng chung dữ liệu nền ứng với cùng data seed. [Phụ lục B của Iannucci et al.](https://arxiv.org/html/2512.19677v1#A2) · [Bộ sinh hiện tại](../../src/simulation/synthetic.py)

| Mô phỏng | Đặc tả dùng trong bộ sinh hiện tại |
| --- | --- |
| 1 | Cửa sổ 15 phút từ phút 1920, timestamp đều có điều kiện trên số sự kiện. Mỗi tài khoản phối hợp xuất hiện ít nhất một lần, các sự kiện còn lại được gán ngẫu nhiên. |
| 2 | Ba đợt, mỗi đợt 6 phút và 6 sự kiện, cách nhau 90 phút không hoạt động. Đợt đầu bắt đầu ở phút 1920. Mỗi đợt có đủ 6 tài khoản phối hợp. Timestamp đều trong từng đợt. |
| 3 | Bốn đợt bắt đầu tại phút 1920, 4860, 7710 và 8430, kéo dài 5, 5, 3 và 2 phút, với 5, 6, 4 và 3 sự kiện. Ba đợt đầu lần lượt dùng nhóm `(00, 01)`, `(02, 03)`, `(04, 05)`. Đợt cuối dùng nhóm `(00, 02, 04)`. Timestamp đều trong mỗi đợt. |

Poisson cắt ngưỡng ở mô phỏng 1 là giả định triển khai dựa trên mô hình đếm của quá trình Poisson đồng nhất trong một cửa sổ. Với tốc độ danh nghĩa 1,3 sự kiện/phút và cửa sổ chọn dài 15 phút, tham số trước điều kiện hóa là `λ = 19,5`. Lấy mẫu lại cho đến khi `15 ≤ N ≤ 20` thực thi giới hạn tổng đã nêu trong Phụ lục B và vẫn cho phép số sự kiện dao động. Đây là lấy mẫu từ phân phối có điều kiện. Kỳ vọng sau điều kiện hóa xấp xỉ 17,726 sự kiện, tương đương 1,182 sự kiện/phút trên cửa sổ đã chọn. Phụ lục B cho tốc độ trung bình và giới hạn số sự kiện, còn luật đếm, lịch đợt cụ thể, ánh xạ người dùng, cách chia danh mục và seed thuộc phần tái dựng. [Phụ lục B của Iannucci et al.](https://arxiv.org/html/2512.19677v1#A2)

## Phiên bản, kiểm thử và tái lập

- Data seed `20261108`–`20261137` và community seed `42`–`46`. Bảy cấu hình gồm `1`, `2`, `3`, `1+2`, `1+3`, `2+3` và `1+2+3`.
- Bản chẩn đoán dựa trên HEAD `0db9c18812928b93507a122a0ee6275737cd3a03` cùng mã trong working tree. Git của mã tham chiếu là `bae7b15b9e29e2c00dde585c5eaf210c10092730`. SHA-256 của nguồn thực thi, cấu hình và hai CSV lịch sử đã được ghi trong [run_metadata.json](diagnostics-2026-11-18/run_metadata.json).
- Chẩn đoán mất **380,281 giây wall-clock** trên Windows 11 và Python 3.13.15, với NumPy 2.5.4, pandas 3.1.1, NetworkX 3.7, SciPy 1.18.1, igraph 1.0.0, leidenalg 0.12.0, scikit-learn 1.9.1 và pytest 9.2.0.
- Hai CSV lịch sử và các tệp theo seed thiếu manifest ghi commit, hash working tree, phiên bản thư viện và runtime của lần chạy cũ. SHA-256 của runner trước khi tách helper là `ef259e68dface3d44792ebc97b7f99d39ae9059ed4a7303a1bfed2b346c7b8d9`. Hai helper được tách từ cùng mã làm tròn và tạo danh sách cạnh. Dấu thời gian tệp cho thấy khoảng 24 phút 40 giây giữa lúc ghi kết quả seed đầu và `results_all.csv`. Đây là khoảng cách giữa hai lần ghi, chưa tính thời gian chạy seed đầu.
- Baseline `130 passed`. Bộ đầy đủ sau khi bổ sung có `139 passed` trong 53,42 giây, gồm 108 test dự án và 31 test tham chiếu. Chín test bổ sung kiểm tra tie theo chiều canonical, cùng β và chuẩn hóa theo nội dung, lọc cạnh bằng ngưỡng nghiêm ngặt sau khi cộng trọng số, sai khác trên hợp tập cạnh, tích seed đầy đủ, loại bản sao greedy và lặp lại chẩn đoán cho cả ba mô phỏng của một seed.

Chạy tại thư mục gốc dự án bằng PowerShell. Dùng một thư mục đầu ra mới cho mỗi lần.

```powershell
$env:PYTHONPATH = 'references/time-aware-collaboration'
$env:PYTHONHASHSEED = '0'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TQDM_DISABLE = '1'
.\.venv-kltn\Scripts\python.exe -m scripts.diagnose_t6b --output-dir reports/t6b/diagnostics-repeat
.\.venv-kltn\Scripts\python.exe -m pytest -p no:cacheprovider tests references/time-aware-collaboration/tests
```

## Hạn chế và phạm vi kết luận

Tiêu chí U4 của review xét F1* ở cấu hình 3. Phụ lục B báo cáo Homogeneity và WP đều bằng 1,00 ở cấu hình này. Kết quả T6b đánh giá độ phù hợp của F1* theo phân phối trên bộ sinh hiện có. Các chênh lệch ở hai chỉ số còn lại, lịch đợt tự chọn, phụ thuộc giữa các lần lặp dùng cùng dữ liệu và thiếu manifest lịch sử giới hạn khả năng tái lập chính xác từng ô của Bảng 2. [Phụ lục B và Bảng 2](https://arxiv.org/html/2512.19677v1#A2) · [Review 03 §6.4](../../reviews/Review-2026-11-15.md#64-chẩn-đoán-t6-cấu-hình-3-t6b-tối-đa-2-ngày)

Kết luận kỹ thuật được ghi tại [nhật ký T6b](../../docs/tasks.md#evidence-log-append-as-tasks-progress). Xác nhận của GVHD được theo dõi riêng.
