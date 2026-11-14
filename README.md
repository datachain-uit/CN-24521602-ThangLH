
# Tái lập Bảng 2 – Phụ lục B (Iannucci et al., 2026)

Mã nguồn tái dựng thí nghiệm trên dữ liệu mô phỏng trong bài báo *Detecting Coordinated Activities Through Temporal, Multiplex, and Collaborative Analysis* (ICWSM 2026).

Quy trình gồm:
1. Sinh ba bộ dữ liệu mô phỏng.
2. Xây dựng đồ thị phối hợp có trọng số thời gian theo Eq. 2.
3. Tìm tham số β tối ưu bằng modularity.
4. Phát hiện cộng đồng bằng thuật toán Leiden.
5. Đánh giá bằng F1*, Homogeneity và Weighted Precision.

## 1. Cài đặt

Chạy tại thư mục gốc dự án:

```bash
python -m pip install -r requirements.txt
```

## 2. Chạy tái lập Bảng 2

```bash
python -m src.eval.reproduce_table2
```

Chạy lệnh từ **thư mục gốc dự án**.

Chương trình thực hiện tìm kiếm β từ 0 đến 10, bước 0.01, rồi đánh giá bảy cấu hình:

- Simulation 1
- Simulation 2
- Simulation 3
- Simulation 1 + 2
- Simulation 1 + 3
- Simulation 2 + 3
- Simulation 1 + 2 + 3

## 3. Kết quả đầu ra

Các file kết quả được lưu trong `reports/`:

| File | Nội dung |
|---|---|
| `t6_simulation1_beta_grid.csv` | Modularity Q theo β của Simulation 1 |
| `t6_simulation2_beta_grid.csv` | Modularity Q theo β của Simulation 2 |
| `t6_simulation3_beta_grid.csv` | Modularity Q theo β của Simulation 3 |
| `t6_optimal_betas.csv` | β tối ưu, modularity Q, số đỉnh và số cạnh |
| `t6_table2_reproduced.csv` | F1*, Homogeneity và Weighted Precision của 7 cấu hình |

## 4. Cấu trúc mã nguồn

```text
src/
├── simulation/
│   └── synthetic.py
├── graph/
│   └── time_aware.py
├── community/
│   ├── modularity.py
│   └── leiden.py
└── eval/
    ├── metrics.py
    └── reproduce_table2.py
```

## 5. Lưu ý về mức độ tái lập

Đây là bản tái dựng dựa trên nội dung được công bố trong bài báo, chưa được xác nhận là tái lập chính xác Bảng 2.

Một số chi tiết chưa được bài báo quy định đầy đủ:

- Seed ngẫu nhiên.
- Thời gian và cách phân bổ sự kiện cụ thể trong các đợt hoạt động.
- Cách ánh xạ các loại hành động vào layer của Bảng 2.

Những lựa chọn bổ sung trong mã nguồn được xem là giả định triển khai, không phải thông số đã được tác giả xác nhận.

## 6. Tài liệu tham khảo

Iannucci, L., Muratore, E., Matakos, A., & Kivelä, M. (2026).

*Detecting Coordinated Activities Through Temporal, Multiplex, and Collaborative Analysis.*

International AAAI Conference on Web and Social Media (ICWSM 2026).

Các phần được sử dụng: Eq. 2, Mục 4.1, Hình 2, Hình 5, Hình 6, Phụ lục A và B.
