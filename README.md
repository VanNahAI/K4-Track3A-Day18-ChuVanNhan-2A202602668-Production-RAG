# Lab 18: Production RAG Pipeline

**K4-Track3A · Ngày 18 · Production RAG**  
**Thời gian:** 2h implement + 30 phút reflection

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

Xem **ASSIGNMENT.md** để biết chi tiết từng module và timeline.

## Prerequisites

| Dependency | Bắt buộc? | Dùng cho |
|-----------|-----------|----------|
| Docker (Qdrant) | ✅ Có | M2 Dense Search |
| Python 3.11 | ✅ Có | Tất cả modules; dùng phiên bản trong `.python-version` để tương thích dependency hiện tại |
| `OLLAMA_API_KEY` | ⚠️ LLM/M4/M5 | Ollama Cloud: trả lời, RAGAS eval, enrichment |

**Pre-download models** (tránh timeout trong lab):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

## Quick Start

### 1. Clone repository & tạo môi trường ảo

**Linux / macOS / Git Bash:**
```bash
git clone <repo-url>
cd K4-Track3A-Production-RAG
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd K4-Track3A-Production-RAG
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
*(Nếu dùng Windows CMD: chạy `.venv\Scripts\activate.bat`)*

Nếu không có lệnh `py`, dùng đường dẫn đầy đủ tới Python 3.11 để chạy `-m venv .venv`.
Nếu môi trường ảo cũ dùng Python 3.13/3.14, đổi tên để giữ bản sao rồi tạo lại bằng Python 3.11; không tái sử dụng môi trường cũ.
Lỗi NumPy 1.26.4 `Unknown compiler(s)` trên Python 3.13/3.14 xảy ra vì không có wheel tương thích và pip chuyển sang build từ source. Dùng Python 3.11 với bộ dependency hiện tại, không cài compiler để xử lý lỗi này.

### 2. Cài đặt dependencies & Khởi động dịch vụ

**Linux / macOS / Git Bash:**
```bash
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
cp .env.example .env                    # Tạo file .env và điền OLLAMA_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```

**Windows (PowerShell):**
```powershell
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
Copy-Item .env.example .env             # Tạo file .env và điền OLLAMA_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```
*(Nếu dùng Windows CMD: dùng `copy .env.example .env` thay cho `Copy-Item`)*

## Ollama Cloud

Tạo API key tại https://ollama.com/settings/keys, sao chép `.env.example` sang `.env`, rồi điền `OLLAMA_API_KEY` trong máy cá nhân (không commit hoặc gửi key qua chat).

```dotenv
OLLAMA_API_KEY=<your-ollama-api-key>
OPENAI_BASE_URL=https://ollama.com/v1
LLM_MODEL=gpt-oss:120b-cloud
```

Pipeline, baseline và enrichment dùng OpenAI SDK với endpoint Ollama Cloud. RAGAS dùng cùng chat model và embeddings local `all-MiniLM-L6-v2`, không gọi OpenAI embeddings. Không cần cài Ollama local. Chạy cloud sẽ gửi câu hỏi, contexts và văn bản tới Ollama; chỉ dùng dữ liệu được phép chia sẻ. Muốn quay lại OpenAI: đặt `OPENAI_BASE_URL=https://api.openai.com/v1`, `LLM_MODEL=gpt-4o-mini`, bỏ `OLLAMA_API_KEY` và đặt `OPENAI_API_KEY`.

## Module 1: Chunking

- `chunk_semantic()`: tách câu bằng regex, mã hóa bằng `all-MiniLM-L6-v2`, ngắt khi cosine giữa hai câu liên tiếp nhỏ hơn `SEMANTIC_THRESHOLD` (0.85). Model được tải một lần; lần đầu cần tải model nếu chưa có cache.
- `chunk_hierarchical()`: gom đoạn văn thành cha tối đa 2048 ký tự, chia con tối đa 256 ký tự; đoạn quá dài được ngắt tại khoảng trắng, từ quá dài bị cắt để giữ giới hạn. `parent_id` nằm trong metadata của cha/con và thuộc tính của con.
- `chunk_structure_aware()`: tách theo tiêu đề Markdown cấp 1–3, lưu tên trong `section`; giữ bảng, danh sách và fenced code block trong cùng phần. Văn bản trước tiêu đề có `section` rỗng.

Kiểm tra: `python -m pytest tests/test_m1.py -q`.

## Module 2: Hybrid Search

- BM25: dùng `underthesea`, đổi `_` thành khoảng trắng, chuyển cả tài liệu và câu hỏi thành chữ thường; chỉ trả kết quả có điểm dương.
- Dense: dùng `BAAI/bge-m3` (1024 chiều), lưu vector cosine và metadata vào Qdrant; truy vấn bằng `query_points()`. `index()` thay thế toàn bộ collection được chỉ định sau khi mã hóa và kiểm tra kích thước vector; không dùng collection chứa dữ liệu cần giữ. Nếu Qdrant không kết nối được lúc khởi tạo, fallback hiện tại dùng RAM, dữ liệu không tồn tại sau khi kết thúc tiến trình.
- RRF: cộng `1 / (RRF_K + rank + 1)`, mặc định `RRF_K = 60`; gộp theo nội dung `text`, giữ metadata từ kết quả đầu tiên, không cộng trùng trong cùng danh sách.

Kiểm tra: `python -m pytest tests/test_m2.py -v`. Test Dense dùng Qdrant trong RAM và encoder giả lập; không tải model lớn hoặc cần Docker.

## Module 3: Cross-Encoder Reranking

`CrossEncoderReranker` nạp `BAAI/bge-reranker-v2-m3` bằng `sentence_transformers.CrossEncoder` khi cần và tái sử dụng model. Mỗi cặp `(query, text)` được chấm bằng `predict()`, sắp xếp giảm dần, giữ mặc định 3 kết quả (`RERANK_TOP_K`). Kết quả giữ điểm truy xuất ban đầu, metadata và `rank` bắt đầu từ 0. Danh sách rỗng hoặc `top_k=0` không nạp model.

Kiểm tra: `python -m pytest tests/test_m3.py -v`. Test dùng model giả lập để kiểm tra xếp hạng và xử lý biên, không đánh giá chất lượng model thật. `FlashrankReranker` vẫn là phần tùy chọn chưa triển khai; độ trễ cần đo trên phần cứng thực tế, không đảm bảo dưới 5ms.

## Module 4: RAGAS Evaluation

`evaluate_ragas()` tạo `Dataset` với câu hỏi, câu trả lời, contexts và ground truth; tính `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, trả điểm trung bình và `per_question` dạng `EvalResult`. Dữ liệu đầu vào phải cùng số phần tử. RAGAS chạy tối đa 2 tác vụ đồng thời, timeout 300 giây mỗi metric và tối đa 1 lần thử theo RunConfig. Metric lỗi không hủy toàn bộ lượt đánh giá: trung bình chỉ tính các điểm hợp lệ, `metric_counts` lưu số điểm mỗi metric và `failed_questions` lưu câu hỏi/metric lỗi. `per_question` chỉ chứa câu hỏi đủ 4 điểm hợp lệ. Kết quả thiếu có trường `error`; `main.py` không so sánh baseline/production khi một bên chưa hoàn tất. Lỗi toàn lượt vẫn trả điểm 0 kèm `error`, không phải điểm đánh giá thật.

`failure_analysis()` lấy câu hỏi có điểm trung bình thấp nhất, tìm metric thấp nhất và gợi ý sửa prompt, chunking/BM25 hoặc reranking/metadata. `score` là trung bình 4 metric; khi hòa, ưu tiên theo thứ tự faithfulness, answer_relevancy, context_precision, context_recall.

Kiểm tra: `python -m pytest tests/test_m4.py -v`. Test giả lập RAGAS, không gọi API. Đánh giá thật mặc định dùng Ollama Cloud, cần `OLLAMA_API_KEY` và gửi dữ liệu đánh giá tới dịch vụ; chỉ chạy với dữ liệu được phép chia sẻ.

## Module 5: Enrichment

Chế độ mặc định `enrich_chunks()` gọi model trong `LLM_MODEL` (`gpt-oss:120b-cloud`) một lần mỗi chunk để lấy JSON gồm `summary`, `questions`, `context`, `metadata`. Context được thêm trước văn bản gốc, câu hỏi HyQA thêm phía sau để M2 thực sự index chúng; metadata gốc luôn được ưu tiên giữ nguyên. Chỉ có tên nguồn và chunk được gửi, không có toàn bộ tài liệu, nên context không suy luận được vị trí chính xác trong tài liệu.

Thiếu API key, lỗi API hoặc JSON không hợp lệ: fallback lấy hai câu đầu làm summary, biến câu thành câu hỏi đơn giản, thêm tên nguồn và metadata mặc định (không phải metadata được suy luận). Các hàm riêng dùng chung logic này. API có timeout 30 giây, không tự retry để tránh nhiều request mỗi chunk. Enrichment có ngân sách 2048 token, dùng `reasoning_effort=low` với GPT-OSS để giảm phần reasoning; phản hồi chưa kết thúc hoặc JSON lỗi được fallback và log `finish_reason`/độ dài, không in nội dung tài liệu.

Kiểm tra: `python -m pytest tests/test_m5.py -v`. Test dùng fallback và OpenAI giả lập, không gửi dữ liệu ra ngoài. Chạy thật với API key sẽ gửi nội dung tới Ollama Cloud; chỉ dùng dữ liệu được phép chia sẻ và kiểm tra hạn mức/tính phí của tài khoản.

## Chạy toàn bộ & Kiểm tra

```bash
python main.py                          # Chạy Naive + Production + In bảng so sánh
python check_lab.py                     # Script kiểm tra hợp lệ trước khi nộp (chạy được trên mọi OS)
```

## Cấu trúc repo

```
K4-Track3A-Production-RAG/
├── README.md                   # File này
├── ASSIGNMENT.md               # ★ Đề bài + timeline + reflection
├── RUBRIC.md                   # Hệ thống chấm điểm
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Corpus tiếng Việt — 25 .md files + 3 PDFs (28 files total)
│   ├── nghi_phep_nam_v2023.md  # Nghỉ phép 12 ngày (v2023, superseded)
│   ├── nghi_phep_nam_v2024.md  # Nghỉ phép 15 ngày (v2024, hiện hành)
│   ├── mat_khau_v1.md          # Password policy 90 ngày (OLD)
│   ├── mat_khau_v2.md          # Password policy 120 ngày + MFA (NEW)
│   ├── ... (28 files total)    # 8 categories: leave, salary, IT, workflow, training, admin, safety, compliance
│   ├── so_tay_an_toan.pdf      # An toàn PCCC + sơ cứu (PDF text)
│   ├── BCTC.pdf                # Báo cáo tài chính (scan, cần OCR)
│   └── Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf # Nghị định BVDL (scan, cần OCR)
├── test_set.json               # 20 Q&A pairs (6 types: lookup, version, negation, multi-hop, numeric, ambiguous)
│
├── src/                        # ★ Scaffold code (có TODO markers)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   ├── m5_enrichment.py        # Module 5: Enrichment Pipeline
│   └── pipeline.py             # Ghép toàn bộ pipeline
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (cá nhân)
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (bắt buộc: reports/ragas_report.json)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    └── failure_analysis.md
```

## Timeline (Thời lượng ước tính)

| Thời lượng | Hoạt động |
|------------|-----------|
| 10 phút | Setup môi trường + chạy `naive_baseline.py` |
| 90 phút | Implement M1 → M2 → M3 → M4 → M5 |
| 20 phút | Chạy pipeline + RAGAS + failure analysis |
| 30 phút | Reflection: lecture mapping + project plan |

## Quy chuẩn đặt tên Repository & Nộp bài

- **Cấu trúc đặt tên repo:**  
  `K4-Track3A-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`  
  *(Ví dụ: `K4-Track3A-DAY18-NguyenVanAn-AI20K001-ProductionRAG`)*
- **Hạn chót nộp bài:** **23h59 ngày diễn ra bài lab (GMT+7)** trên cổng VLearn LMS / Codelab.
- **Chi tiết yêu cầu:** Xem tại [ASSIGNMENT.md](ASSIGNMENT.md) và [RUBRIC.md](RUBRIC.md).
