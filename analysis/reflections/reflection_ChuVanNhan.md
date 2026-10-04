# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Chu Văn Nhân

**Khóa:** K4 - Track 3A

**Ngày hoàn thành:** 04/10/2026

Bản tổng hợp dựa trên code, log và report thực tế; kế hoạch bên dưới là đề xuất cho project lab, không khẳng định đã triển khai ngoài phạm vi này.

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | So cosine giữa câu liên tiếp, ngắt khi dưới 0.85; chưa có số liệu so số chunk với baseline. Pipeline thực tế dùng hierarchical 2048/256 ký tự. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 chuẩn hóa `_` và chữ thường; dense dùng bge-m3 1024 chiều. RRF cộng thứ hạng với k=60, không chuẩn hóa điểm hai bộ tìm kiếm. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 chấm cặp query/text, lấy top 3. Chưa lưu latency độc lập nên không khẳng định đạt dưới 5ms. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Report Production đủ 20 câu mỗi metric: faithfulness 0.8050, relevancy 0.6183, precision 0.7958, recall 0.7917. |
| Contextual embeddings | M5 | `_enrich_single_call()` | Một request lấy summary, HyQA, context và metadata; context thêm trước chunk, HyQA thêm vào nội dung index. Fallback local không tương đương hiểu ngữ cảnh bằng LLM. |

So với baseline, faithfulness tăng 0.0336 và relevancy tăng 0.3715; precision giảm 0.1125, recall giảm 0.0333. Cần ablation để xác định đóng góp từng module; không kết luận mọi kỹ thuật đều cải thiện mọi metric.

## Phần 2: Khó khăn & Cách giải quyết

- **`metadata-generation-failed` / NumPy 1.26.4 `Unknown compiler(s)`:** môi trường dùng Python 3.14, trong khi project chọn 3.11. Tạo lại venv bằng Python 3.11.16; kiểm tra import NumPy và `pip check` thành công.
- **`Enrichment API failed (JSONDecodeError); using local fallback`:** nội dung không parse được JSON. Chưa lưu response nên không xác nhận chính xác nguyên nhân; tăng ngân sách 600 lên 2048 token, giảm reasoning GPT-OSS và log finish reason/độ dài mà không in nội dung nhạy cảm.
- **`RAGAS evaluation failed (TimeoutError)`:** chế độ fail-fast làm mất điểm toàn lượt; log có coroutine chưa được await. Chuyển sang xử lý lỗi từng metric, giảm concurrency xuống 2, timeout 300 giây, giữ điểm hợp lệ và ghi số metric thành công. Report mới đủ 20 điểm mỗi metric và không có failed_questions; chưa chứng minh một thay đổi riêng lẻ giải quyết toàn bộ vấn đề.
- **Giới hạn quan sát:** report chưa lưu answer/contexts từng câu. Worst metric chỉ định hướng điều tra, không chứng minh hallucination hoặc lỗi retrieval cụ thể.
- **Kiến thức cần bổ sung:** giới hạn token của reasoning model, API tương thích OpenAI, semantics timeout/retry, cách đánh giá câu hỏi nhiều bước và quản lý phiên bản tài liệu. Kiểm tra tài liệu thư viện theo phiên bản cài đặt và dùng trace nhỏ trước khi chạy toàn bộ.

## Phần 3: Action Plan cho Project cá nhân

### Project: Production RAG tra cứu chính sách nội bộ (project lab)

#### 1. Hiện trạng

Pipeline: hierarchical chunking, enrichment, BM25 + dense + RRF, Cross-Encoder top 3, LLM trả lời, RAGAS. Bottleneck đã thấy: relevancy dưới 0.75; precision/recall giảm so baseline; câu hỏi nhiều bước cần nhiều nguồn. Metadata có parent_id nhưng pipeline hiện chưa mở rộng child thành parent khi trả lời; chưa có bộ lọc phiên bản hiện hành.

#### 2. Kế hoạch cải tiến

1. **Chunking:** giữ bảng Markdown và mở rộng parent khi trả lời; đo tác động với kích thước context/token cố định.
2. **Retrieval:** giữ Hybrid + RRF; thử tách câu hỏi nhiều bước và lọc effective version dựa trên metadata có bằng chứng, không chỉ đoán từ tên file.
3. **Reranking:** giữ Cross-Encoder; đo latency warm/cold và kiểm tra top 3 có đủ nguồn trước khi tăng số context.
4. **Evaluation:** lưu trace answer/contexts/source cùng metric, kiểm tra bottom 5 và chạy lại nhiều lượt để đánh giá độ ổn định. Không so sánh lượt thiếu metric như lượt hoàn tất.
5. **Enrichment:** thử có/không HyQA, cache kết quả theo nội dung/model để giảm chi phí; giữ nguyên source/version metadata và không index suy đoán như dữ kiện gốc.

#### 3. Timeline triển khai

- **Tuần 1:** bổ sung trace, kiểm tra bottom 5, triển khai metadata phiên bản và kiểm thử truy xuất parent.
- **Tuần 2:** chạy ablation enrichment/retrieval, đo latency và token, chọn cấu hình dựa trên 4 metric thay vì chỉ tổng điểm.

## Nguồn

- [Failure analysis](../failure_analysis.md)
- [Production report](../../reports/ragas_report.json)
- [Naive report](../../reports/naive_baseline_report.json)
