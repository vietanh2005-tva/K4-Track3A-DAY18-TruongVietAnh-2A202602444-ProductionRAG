# Individual Reflection — Lab 18: Production RAG

Học viên: Trương Việt Anh 
MSSV: 2A202602444 
Khóa K4, Track 3A

Ngày cập nhật: 04/10/2026.

| Metric | Baseline | Production |
|---|---:|---:|
| Faithfulness | 0.8833 | 0.9429 |
| Answer Relevancy | 0.7416 | 0.9040 |
| Context Precision | 0.7000 | 0.9000 |
| Context Recall | 0.8500 | 0.9333 |

Cùng Gemini 3.1 Flash Lite và embedding bge-m3 cho hai lần đánh giá. Đủ điều kiện bốn tiêu chí bonus (10 điểm theo công thức lab), không thay thế quyết định chấm của giảng viên. Kiểm tra cuối đạt 37/37 test, dùng model thật và tắt API trong unit test; chất lượng API được đánh giá riêng qua RAGAS thật.

## Phần 1: Lecture Mapping

| Khái niệm | Module và hàm | Áp dụng trong bài làm |
|---|---|---|
| Semantic chunking | M1 — `chunk_semantic()` | Tách câu, mã hóa bằng all-MiniLM-L6-v2, so sánh cosine giữa hai câu liên tiếp; ngắt nhóm dưới ngưỡng 0.85. Model được tái sử dụng để tránh tải lại cho từng văn bản. |
| Parent–Child chunking | M1 — `chunk_hierarchical()`; pipeline — `run_query()` | Parent tối đa 2048 ký tự, child tối đa 256. Pipeline cấp ID riêng theo tài liệu, tìm trên child và lấy lại parent gốc để trả lời có đủ bối cảnh. |
| Structure-Aware chunking | M1 — `chunk_structure_aware()` | Tách theo tiêu đề Markdown, giữ section trong metadata. Không xem dấu # bên trong fenced code block là tiêu đề. |
| BM25 + Dense Search | M2 — `segment_vietnamese()`, `BM25Search`, `DenseSearch` | Chuẩn hóa dấu gạch dưới của underthesea, áp dụng cùng cách tách từ cho query và corpus; vector bge-m3 1024 chiều được lưu trong Qdrant. |
| Reciprocal Rank Fusion | M2 — `reciprocal_rank_fusion()` | Cộng `1/(60+rank+1)` theo thứ hạng, tránh phải chuẩn hóa thang điểm BM25 và cosine. |
| Cross-Encoder reranking | M3 — `CrossEncoderReranker.rerank()` | Chấm đồng thời cặp query–document bằng bge-reranker-v2-m3; sắp giảm dần và lấy top 3, giữ điểm gốc và metadata. |
| RAGAS và Diagnostic Tree | M4 — `evaluate_ragas()`, `failure_analysis()` | Đo Faithfulness, Answer Relevancy, Context Precision và Context Recall; dùng metric thấp nhất để gợi ý nguyên nhân rồi kiểm tra lại bằng đáp án và ngữ cảnh thật. Trạng thái failed/skipped không được coi là điểm đo thật. |
| Contextual enrichment và HyQA | M5 — `_enrich_single_call()`, `enrich_chunks()` | Một yêu cầu Gemini/chunk tạo summary, questions, context và metadata. Văn bản được index chứa thông tin làm giàu; đoạn gốc và liên kết parent vẫn được giữ. |
| Latency breakdown | `LatencyTracker`, pipeline | Tách thời gian chunking, enrichment, indexing, retrieval, RRF, reranking, context assembly, generation và RAGAS. Tải model được đo riêng với inference. |

## Phần 2: Challenges & Debugging

### Chạy lệnh ở thư mục cha

Các lỗi thực tế: `no configuration file provided: not found`; `Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'`; `Cannot find path ... .env.example because it does not exist`.

Nguyên nhân là đang đứng ở thư mục Day18, trong khi file cấu hình nằm trong repository con. Cách xử lý là chuyển vào repository trước khi chạy Docker Compose, cài thư viện và tạo `.env`. Môi trường ảo dùng cho bài làm cũng nằm trong repository.

### Đổi nhà cung cấp LLM

Đổi khóa trong `.env` chưa đủ vì code cũ cố định `gpt-4o-mini` và endpoint OpenAI. Bài làm đã dùng `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_BASE_URL` và `GEMINI_API_KEY`; baseline, generation, M4 và M5 sử dụng cấu hình này. Kết nối Gemini đã được kiểm tra bằng một câu thử thật.

Một lần kiểm tra từ thư mục khác báo thiếu khóa dù `.env` của lab đã có khóa. `load_dotenv()` được sửa thành đường dẫn tuyệt đối tính từ `config.py`, để không phụ thuộc thư mục gọi chương trình.

### Phân biệt test logic với kết quả thực tế

Khi model còn tải, đã dùng model/API giả lập để kiểm tra mapping dữ liệu, thứ tự top-k, công thức RRF và fallback. Những kiểm thử đó không chứng minh chất lượng của mô hình thật. Điểm benchmark và Bottom-5 phải lấy từ lần chạy pipeline thật trên 20 câu hỏi.

RAGAS cần truyền riêng LLM và embedding khi dùng Gemini: endpoint tương thích OpenAI của Gemini không cung cấp model embedding mang tên OpenAI. Bài làm dùng embedding cục bộ cho bước đánh giá.

Lần chạy thật với `gemini-3.5-flash` bị giới hạn 20 yêu cầu/ngày/project/model nên RAGAS thất bại. Sau khi được cho phép gửi dữ liệu lab, đã kiểm tra và chuyển sang `gemini-3.1-flash-lite` cho cả baseline và production. Model này có lúc trả về 503 do quá tải; đã bổ sung thử lại có giới hạn và checkpoint sau từng câu baseline. Báo cáo failed không được đưa vào so sánh hay xét bonus.

Tệp bge-reranker-v2-m3 khoảng 2.27 GB bị timeout khi tải qua đường mặc định. Đã kiểm tra đường tải trực tiếp chính thức, bổ sung resume và kiểm tra SHA-256 trước khi nạp model; chưa kết luận benchmark khi model chưa tải đủ.

Sau đó model đã tải đủ và SHA-256 khớp bản chính thức. M3 đạt 5/5 test riêng. Lần chạy chung có lỗi Windows 1455 (paging file quá nhỏ), nên đã dùng chung encoder bge-m3 và nạp reranker FP16 thay vì tự thay đổi cấu hình Windows. Lần chạy lại đạt 37/37 unit test: M1/M2/M3 dùng model thật; M4/M5 kiểm thử chức năng không gọi API, còn chất lượng được xác minh qua RAGAS chạy riêng.

Gemini 3.1 Flash Lite từ chối `n=3` trong một request. Adapter giữ nguyên 3 mẫu Answer Relevancy bằng 3 request `n=1`, lưu từng mẫu độc lập để không nhân bản một câu trả lời. Giới hạn thực tế 15 request/phút được xử lý bằng bộ điều tiết SQLite dùng chung giữa các tiến trình, cách nhau ít nhất 4.5 giây. RAGAS dùng bge-m3 đa ngôn ngữ, dùng chung encoder với retrieval, nhất quán cho baseline và production.

Benchmark 20 cặp đại diện, 5 lần chạy sau warm-up: BGE FP16 trên CPU trung bình 20664.83 ms, chưa đạt 150 ms. Flashrank TinyBERT ONNX đạt trung bình 20.77 ms với cùng cặp đầu vào. Pipeline chính vẫn dùng BGE; phương án nhẹ chỉ được kết luận nhanh hơn, chưa kết luận chất lượng tiếng Việt tốt hơn. Độ trễ end-to-end trên 20 query được lấy riêng từ latency report.

Đo bổ sung BGE FP32 đạt trung bình 2968.72 ms với cùng 20 cặp, nhanh hơn FP16 trên CPU này nhưng vẫn chưa đạt 150 ms. Cấu hình local chuyển sang FP32; test và pipeline chạy tuần tự để tránh vượt giới hạn bộ nhớ. Enrichment, mẫu chấm RAGAS và câu trả lời được checkpoint/cache để không gọi API lại khi tiến trình bị gián đoạn. Metadata báo cáo phân biệt cache hit và fallback với yêu cầu API mới.

### Kiểm soát bộ nhớ và đọc lại Bottom-5

FP32 vẫn không thể giữ đồng thời hai model lớn trên máy 16 GB RAM. Pipeline chuẩn bị embedding của 20 query trước, giải phóng encoder trước reranking, rồi giải phóng reranker trước khi nạp embedding chấm RAGAS. Không chạy test nặng song song với pipeline. Lần chạy cuối dùng lại 100 enrichment và 20 câu trả lời hợp lệ trong cache; latency report ghi rõ cache hit, không đại diện độ trễ cold/API mới.

Bottom-5 cho thấy câu Senior 9 năm trả đúng 18 ngày phép nhưng thiếu lương 20–35 triệu. Kiểm tra corpus xác nhận `bang_luong_2024.md` có dữ liệu, nhưng retrieval không đưa vào context; cần ưu tiên multi-hop và đa dạng bằng chứng thay vì thêm tài liệu đã tồn tại. Một số câu trả lời đúng vẫn có metric thấp, nên phải đọc claim, ground truth và context; không mặc định score thấp là hallucination. Phân tích AI ban đầu đã nhầm thiếu context với thiếu corpus và được sửa sau đối chiếu nguồn.

### Dữ liệu PDF không có text layer

Loader hiện đọc được 26 tài liệu, bỏ qua `BCTC.pdf` và `Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf` vì chúng là bản scan không có text layer. Điều này có thể làm giảm Context Recall đối với câu hỏi liên quan. Bước cải tiến phù hợp là OCR và kiểm tra bản trích xuất; không tự đoán nội dung bị thiếu.

## Phần 3: Action Plan

### Đề xuất dự án: Chatbot tra cứu quy chế tiếng Việt

Đây là kế hoạch đề xuất để áp dụng kiến thức của lab, không phải tuyên bố rằng dự án đã được triển khai.

Hiện trạng tham chiếu là Naive RAG: chia theo đoạn văn và tìm vector. Các rủi ro cần xử lý gồm thiếu bối cảnh, bỏ sót số hiệu văn bản, nhầm quy chế cũ/mới và câu trả lời không có căn cứ.

| Thời gian | Công việc | Tiêu chí kiểm tra |
|---|---|---|
| Tuần 1, ngày 1–2 | Chuẩn hóa tài liệu, bổ sung source, năm và hiệu lực; lập danh sách PDF cần OCR. | Có thể truy nguyên mỗi đoạn về tài liệu gốc; không mất điều khoản và bảng. |
| Tuần 1, ngày 3–4 | So sánh Structure-Aware và Parent–Child; xây BM25 + bge-m3 + RRF. | Bộ câu hỏi chứa từ khóa, số hiệu và từ đồng nghĩa đều được kiểm tra. |
| Tuần 1, ngày 5–7 | Bổ sung Cross-Encoder top 3 và đo latency khi model đã warm up. | So sánh thứ hạng trước/sau rerank; đặt ngân sách độ trễ theo phần cứng thực tế. |
| Tuần 2, ngày 1–3 | Dùng combined enrichment, cache theo hash đoạn và model để tránh gọi lại. | Giữ nguyên source/parent_id; kiểm tra chi phí và độ chính xác của context sinh ra. |
| Tuần 2, ngày 4–5 | Chạy RAGAS, phân tích Bottom-5 và xử lý lỗi theo Diagnostic Tree. | So sánh đủ bốn metric với baseline; kiểm tra riêng xung đột phiên bản. |
| Tuần 2, ngày 6–7 | Kiểm thử lại, hoàn thiện giao diện có trích dẫn và cơ chế từ chối khi thiếu bằng chứng. | Chỉ triển khai khi kết quả chất lượng và độ trễ đáp ứng yêu cầu sử dụng. |

Mục tiêu thử nghiệm là Faithfulness ≥ 0.85 và cả bốn metric ≥ 0.75; đây là mục tiêu cần đo, không phải kết quả đã đạt.
