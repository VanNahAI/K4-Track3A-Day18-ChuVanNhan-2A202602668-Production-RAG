# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Chu Văn Nhân

**Khóa:** K4 - Track 3A

**Ngày:** 04/10/2026

## Nguồn và giới hạn

Số liệu lấy từ [Production report](../reports/ragas_report.json), [Naive report](../reports/naive_baseline_report.json) và [test set](../test_set.json). Cả hai report có đủ 20 điểm cho từng metric, không có `failed_questions`. Report không lưu câu trả lời và contexts từng câu; do đó không thể xác nhận output sai cụ thể hoặc kết luận nguyên nhân gốc chỉ từ metric. Chẩn đoán bên dưới là giả thuyết do cây lỗi đề xuất, không phải bằng chứng mô hình đã bịa.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7714 | 0.8050 | +0.0336 |
| Answer Relevancy | 0.2469 | 0.6183 | +0.3715 |
| Context Precision | 0.9083 | 0.7958 | -0.1125 |
| Context Recall | 0.8250 | 0.7917 | -0.0333 |

Production cải thiện faithfulness và answer relevancy nhưng giảm precision và recall. Answer relevancy vẫn dưới 0.75. Chưa có ablation hoặc nhiều lượt chạy nên không quy cải thiện cho riêng enrichment, hybrid hay reranking; không khẳng định Production tốt hơn trên mọi tiêu chí.

## Bottom-5 Failures

`score` là trung bình 4 metric, không phải điểm riêng của worst metric.

### #1 — Tạm ứng và phí quá hạn
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn thanh toán 15 ngày, quá hạn 5 ngày; phí 2%/tháng = 300.000 VNĐ/tháng, pro-rata khoảng 50.000 VNĐ cho 5 ngày.
- **Got:** Không được lưu trong report; chưa xác minh.
- **Worst metric:** `answer_relevancy`; trung bình **0.4170**.
- **Error Tree:** Kiểm tra output có tính phí và trả lời số tiền; đối chiếu contexts có thời hạn và mức phí; kiểm tra truy vấn truy xuất đủ hai điều khoản.
- **Root cause:** Chưa xác nhận. Cây chẩn đoán gợi ý câu trả lời lệch trọng tâm; cần phân biệt lỗi prompt với thiếu ngữ cảnh hoặc tính toán.
- **Suggested fix:** Prompt yêu cầu trả số tiền, công thức, thời gian quá hạn và giả định pro-rata; không suy luận mức phí khi context không có.

### #2 — Phân loại thông tin lương
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Bí mật, cấp 3; mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Không được lưu trong report; chưa xác minh.
- **Worst metric:** `answer_relevancy`; trung bình **0.5281**.
- **Error Tree:** Kiểm tra output nêu cấp độ; kiểm tra contexts có quy chế lương và chính sách phân loại dữ liệu; kiểm tra truy xuất nhiều tài liệu.
- **Root cause:** Chưa xác nhận; có thể trả lời chung chung hoặc thiếu một tài liệu liên quan.
- **Suggested fix:** Trả trực tiếp cấp độ rồi dẫn điều khoản; kiểm tra đủ hai nguồn trước khi sửa retrieval.

### #3 — Hoàn trả tài trợ đào tạo
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết 1 năm; nghỉ trước hạn phải hoàn trả 100%, tức 25.000.000 VNĐ.
- **Got:** Không được lưu trong report; chưa xác minh.
- **Worst metric:** `faithfulness`; trung bình **0.5620**.
- **Error Tree:** Đối chiếu số tiền và tỷ lệ hoàn trả trong output với context; kiểm tra điều khoản có quy định 100% thay vì tính theo tháng.
- **Root cause:** Cây chẩn đoán gợi ý thông tin không được context hỗ trợ; chưa đủ bằng chứng xác nhận hallucination.
- **Suggested fix:** Prompt chỉ sử dụng tỷ lệ trong tài liệu, temperature 0; thêm kiểm tra hồi quy cho trường hợp nghỉ trước thời hạn.

### #4 — Phép năm và lương Senior
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** v2024: 15 + 9/3 = 18 ngày; lương Senior P3–P4: 20–35 triệu VNĐ/tháng.
- **Got:** Không được lưu trong report; chưa xác minh.
- **Worst metric:** `context_recall`; trung bình **0.6274**.
- **Error Tree:** Kiểm tra cả hai đáp án; kiểm tra context có chính sách phép v2024 và bảng lương; kiểm tra tài liệu v2023 có lấn át bản hiện hành không.
- **Root cause:** Có dấu hiệu thiếu ngữ cảnh; chưa xác định thiếu chính sách phép, bảng lương hay cả hai.
- **Suggested fix:** Tách truy vấn nhiều bước, hợp nhất nguồn; ưu tiên phiên bản có hiệu lực và giữ nguyên bảng lương khi chunking.

### #5 — Lương thử việc Junior
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Không được lưu trong report; chưa xác minh.
- **Worst metric:** `answer_relevancy`; trung bình **0.6311**.
- **Error Tree:** Kiểm tra output có số tiền; kiểm tra context có mức lương tối đa và tỷ lệ thử việc; kiểm tra phép nhân.
- **Root cause:** Chưa xác nhận; cây lỗi gợi ý trả lời chưa trực tiếp.
- **Suggested fix:** Prompt yêu cầu kết quả và phép tính ngắn, đồng thời dẫn đủ hai dữ kiện.

## Case Study (cho presentation)

**Question chọn phân tích:** Senior 9 năm thâm niên, phép năm và khoảng lương.

1. Report chưa lưu output, nên chưa kết luận đáp án nào sai.
2. Context phải đồng thời có bảng lương và chính sách phép hiện hành; recall thấp là tín hiệu cần kiểm tra bước này.
3. Pipeline hiện không có query rewrite; câu hỏi nhiều bước cần kiểm tra khả năng tìm đủ nguồn của truy vấn gốc.
4. Thử nghiệm tiếp: lưu answer/contexts/source, thử tách câu hỏi và lọc phiên bản, rồi đo lại cùng test set. Không sửa ground truth để tăng điểm.

**Nếu có thêm 1 giờ:** lưu trace từng câu, kiểm tra 5 trường hợp trên, chạy ablation với/không HyQA và so sánh recall/precision trên cùng bộ câu hỏi. Chỉ tuyên bố nguyên nhân khi đã đối chiếu output với nguồn.
