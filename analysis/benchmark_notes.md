# Reranker benchmark — Lab 18

Đo trên CPU, sau warm-up, 5 lần chạy với 20 child đầu tiên (tối đa 256 ký tự), query “Nhân viên được nghỉ phép bao nhiêu ngày?”, trả top 3. Đây là tập cặp đại diện để đo inference, không phải kết quả retrieval hoặc đánh giá chất lượng đầy đủ.

| Phương án | Average | Min | Max | Average < 150 ms |
|---|---:|---:|---:|---|
| BAAI/bge-reranker-v2-m3, FP16 | 20664.83 ms | 19174.62 ms | 21904.44 ms | Không |
| BAAI/bge-reranker-v2-m3, FP32 | 2968.72 ms | 2812.62 ms | 3385.12 ms | Không |
| Flashrank ms-marco-TinyBERT-L-2-v2, ONNX | 20.77 ms | 12.52 ms | 27.75 ms | Có |

Báo cáo gốc: `reports/reranker_benchmark.json`, `reports/reranker_benchmark_float32.json` và `reports/flashrank_benchmark.json`.

Torch hiện là bản CPU và `torch.cuda.is_available()` trả False. Không tự thay đổi bộ nhớ ảo hoặc cài bộ CUDA trên máy người dùng. FP16 giúp giảm bộ nhớ, không đảm bảo tăng tốc trên mọi CPU.

Pipeline chính dùng BGE FP32 theo bài lab, nhanh hơn FP16 trên CPU Intel Core Ultra 7 155H này. Các tác vụ nặng chạy tuần tự để tránh thiếu bộ nhớ. Flashrank được triển khai và kiểm thử như phương án tùy chọn. Không kết luận Flashrank có chất lượng tiếng Việt ngang BGE chỉ từ độ trễ; cần chạy riêng toàn bộ 20 câu và so sánh RAGAS trước khi thay thế trong dự án thật.
