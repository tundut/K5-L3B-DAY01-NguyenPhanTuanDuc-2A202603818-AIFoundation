# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603818
- **Họ và tên:** Nguyễn Phan Tuấn Đức
- **SEED** (in ở ô đầu notebook): 9136

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

Bảng 1. Chất lượng mô hình
| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) |
|---|---:|---:|---:|---:|
| 1. BoW + LogReg | 0.987 | 0.798 | 0.882 | 0.857 |
| 2. BoW + MLP | 0.956 | 0.915 | 0.935 | 0.947 |
| 3a. MiniLM + kNN | 0.881 | 0.947 | 0.913 | 0.950 |
| 3b. MiniLM + LogReg | 0.928 | 0.957 | 0.942 | 0.974 |
| 3c. MiniLM + MLP head | 0.936 | 0.936 | 0.936 | 0.947 |


Bảng 2. Hiệu năng xử lý
| Mô hình | Train (s) | Tin/s (cả lô) | p50 (ms/tin) | p95 (ms/tin) |
|---|---:|---:|---:|---:|
| 1. BoW + LogReg | 4.584 | 11377.000 | 0.051 | 0.107 |
| 2. BoW + MLP | 9.189 | 14466.640 | 0.307 | 2.380 |
| 3a. MiniLM + kNN | 72.392 | 44.568 | 41.183 | 56.051 |
| 3b. MiniLM + LogReg | 72.827 | 50.156 | 36.174 | 50.770 |
| 3c. MiniLM + MLP head | 73.414 | 52.379 | 25.946 | 33.528 |


Bảng 3. Chi phí và đáp ứng yêu cầu độ trễ
| Mô hình | USD / 1k tin | p95 < 50 ms | F1 (test) |
|---|---:|:---:|---:|
| 1. BoW + LogReg | 0.0 | Đạt | 0.882 |
| 2. BoW + MLP | 0.0 | Đạt | 0.935 |
| 3a. MiniLM + kNN | 0.0 | Không đạt | 0.913 |
| 3b. MiniLM + LogReg | 0.0 | Không đạt | 0.942 |
| 3c. MiniLM + MLP head | 0.0 | Đạt | 0.936 |

• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.  
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.  
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.  
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

Dự đoán 1 — LLM: Em dự đoán model lớn sẽ có F1 cao hơn model nhỏ vài phần trăm. Thực tế, cả hai đều đạt F1 = 0 do cả 40 kết quả đều không hợp lệ. Điều này cho thấy cần kiểm tra cách gọi API và xử lý output trước khi đánh giá chất lượng LLM.

Dự đoán 2 — Few-shot: Em nghĩ few-shot sẽ cải thiện F1 khoảng 1–3%. Thực tế, F1 vẫn bằng 0 và chi phí tăng từ $0.00026 lên $0.00084 cho 40 tin. Nguyên nhân có thể nằm ở lỗi xử lý output, nên chưa thể kết luận few-shot không hiệu quả.

Dự đoán 3 — Stress test: Em dự đoán BoW đạt accuracy 70–80% trên tập khó, nhưng thực tế BoW + LogReg chỉ đạt 54.5%. Mô hình phụ thuộc vào từ khóa nên dễ bỏ sót spam được viết tự nhiên hoặc cố tình biến đổi ký tự.

## 4. Phân tích 50 case khó

BoW + LogReg tụt F1 nhiều nhất trong các mô hình truyền thống, từ 0.857 xuống 0.286, giảm 0.571. Nhóm spam viết tự nhiên và né từ khóa gây nhiều khó khăn.

Hai ví dụ:

Case #12: Tin spam mở đầu lịch sự, giả làm người quen. Các từ thông thường khiến mô hình dự đoán ham.

Case #31: Tin spam viết s.e.l.e.c.t.e.d và c.a.s.h để né từ khóa. Việc tách ký tự khiến mô hình khó nhận diện nội dung spam.

Embedding + kNN đạt F1 = 0.700, cao nhất trên tập khó. Các LLM đạt F1 = 0 do lỗi output nên chưa thể đánh giá công bằng khả năng chống prompt injection.

## 5. Thử thách cá nhân

Đề #5 — Embedding: Tạo 4 SMS gồm 2 spam né từ khóa và 2 ham chứa "free"/"win", sau đó so sánh BoW với Embedding + LogReg.

Em tạo 4 tin nhắn và chạy hai mô hình để kiểm tra. Kết quả:

BoW + LogReg: đúng 3/4 tin (75%).

Embedding + LogReg: đúng 2/4 tin (50%).

Cả hai đều bỏ sót tin spam giả xác minh tài khoản.

Embedding còn bỏ sót tin spam giả thông báo giao hàng.

Kết quả cho thấy Embedding không phải lúc nào cũng tốt hơn BoW. Tuy nhiên, phần thử thách chưa chuẩn hóa embedding bằng mu và sd như lúc huấn luyện, nên cần chạy lại để so sánh chính xác.

## 6. Phương án cho sản phẩm thật

Với 10 triệu SMS/ngày và p95 dưới 50 ms, em chọn BoW + MLP làm mô hình chính.

Mô hình đạt F1 = 0.935, p95 = 2.38 ms và xử lý khoảng 14.467 tin/giây khi chạy theo lô. So với Embedding + MLP head (F1 = 0.936, p95 = 33.53 ms), BoW + MLP nhanh hơn đáng kể mà chất lượng trên test thường gần tương đương.

Đánh đổi là khả năng xử lý spam né từ khóa còn hạn chế. Em đề xuất dùng Embedding để kiểm tra bổ sung các tin khó, đồng thời cần kiểm thử tải thực tế trước khi triển khai.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì? | AI hỗ trợ gì? | Kiểm chứng / chỉnh sửa |
|---|---|---|---|
| Phần 4 | Cách viết `call_openai()`? | Gợi ý gọi API và xử lý kết quả. | Chạy kiểm thử hàm. |
| Phần 4 | Cách tìm các case mọi model đều sai? | Gợi ý lọc DataFrame. | Sửa lỗi `names_m`. |
| Phần 5 | Cách so sánh BoW và Embedding? | Gợi ý 4 SMS và code kiểm thử. | Thay tên biến và chạy dự đoán. |
| Báo cáo | Cách phân tích kết quả benchmark? | Gợi ý so sánh F1, độ trễ và chi phí. | Đối chiếu số liệu notebook. |

**Điều em học được:** Accuracy cao chưa chắc mô hình phát hiện spam tốt. Em cũng nhận ra mô hình phức tạp không phải lúc nào cũng chính xác hơn, và khi triển khai thực tế cần cân bằng F1, độ trễ, chi phí và khả năng xử lý những trường hợp bất thường.