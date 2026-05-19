Bạn là Senior AI Engineer + Backend Architect + RAG Engineer. 
Hãy giúp tôi xây dựng một dự án portfolio tên là:

MedTutor Multimodal RAG Copilot

Mục tiêu:
Xây dựng một chatbot/voicebot hỗ trợ học tập và tra cứu tài liệu y khoa. Hệ thống có thể đọc tài liệu text/PDF/DOCX/HTML, xử lý ảnh y tế dạng học tập bằng OCR/caption, xử lý audio/video transcript, áp dụng nhiều phương pháp RAG khác nhau, trả lời có dẫn nguồn, có guardrail an toàn y tế, có evaluation và deploy được bằng Docker.

Lưu ý cực kỳ quan trọng về phạm vi:
- Đây là dự án Medical Education / Medical Learning Assistant.
- Không được định vị là bot chẩn đoán bệnh.
- Không được kê đơn thuốc.
- Không đưa ra quyết định điều trị cá nhân.
- Mọi câu trả lời y tế phải có disclaimer: “Thông tin chỉ phục vụ học tập/tham khảo, không thay thế tư vấn của bác sĩ.”
- Nếu người dùng hỏi triệu chứng nguy hiểm, cấp cứu, đau ngực, đột quỵ, khó thở, mất ý thức, chảy máu nặng, tự tử hoặc tình huống nguy cấp, bot phải khuyến nghị liên hệ cấp cứu/cơ sở y tế ngay.
- Không lưu dữ liệu bệnh nhân thật trong repo. Chỉ dùng dữ liệu mẫu, giả lập hoặc tài liệu public.

Tôi muốn bạn làm việc theo phong cách:
1. Đầu tiên hãy inspect repo hiện tại nếu có.
2. Nếu repo trống, hãy scaffold dự án từ đầu.
3. Không viết code lộn xộn trong một file duy nhất.
4. Thiết kế kiến trúc rõ ràng, module hóa, dễ mở rộng.
5. Mỗi phase phải có code chạy được, test được.
6. Ưu tiên MVP chạy được trước, sau đó mở rộng advanced RAG.
7. Viết README chi tiết để tôi có thể đưa vào portfolio/GitHub.
8. Không hard-code API key. Dùng .env và .env.example.
9. Với các phần cần API như LLM, embedding, speech-to-text, image captioning, hãy tạo abstraction/interface để có thể thay provider.
10. Nếu thiếu API key, hệ thống vẫn chạy được ở chế độ mock/local demo.

Tech stack ưu tiên:
- Python 3.11+
- FastAPI cho backend API
- Streamlit cho frontend demo nhanh
- Qdrant làm vector database
- SQLite cho MVP metadata DB, thiết kế sao cho sau này đổi sang PostgreSQL được
- Docker + docker-compose
- Pydantic cho schema/config
- Pytest cho test
- Ruff/Black hoặc cấu hình format cơ bản
- LangChain hoặc LlamaIndex đều được, nhưng code phải rõ ràng, không phụ thuộc quá nặng vào framework
- Có thể dùng sentence-transformers cho embedding local nếu không có API key
- Có thể dùng rank-bm25 cho BM25
- Có thể dùng faster-whisper/openai whisper abstraction cho STT
- OCR có thể dùng pytesseract hoặc easyocr abstraction
- Với image captioning, tạo interface và mock trước; sau này thay bằng vision model

Yêu cầu kiến trúc thư mục:

medtutor-rag-copilot/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── chat.py
│   │   ├── upload.py
│   │   ├── voice.py
│   │   ├── ingest.py
│   │   └── health.py
│   ├── schemas/
│   │   ├── chat.py
│   │   ├── document.py
│   │   ├── retrieval.py
│   │   └── evaluation.py
│   ├── ingestion/
│   │   ├── base.py
│   │   ├── text_loader.py
│   │   ├── pdf_loader.py
│   │   ├── docx_loader.py
│   │   ├── html_loader.py
│   │   ├── image_processor.py
│   │   ├── audio_processor.py
│   │   ├── video_processor.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   ├── rag/
│   │   ├── naive_rag.py
│   │   ├── hybrid_rag.py
│   │   ├── parent_child_rag.py
│   │   ├── multi_query_rag.py
│   │   ├── rag_fusion.py
│   │   ├── rerank_rag.py
│   │   ├── corrective_rag.py
│   │   ├── agentic_router.py
│   │   └── prompts.py
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   ├── bm25_store.py
│   │   ├── metadata_store.py
│   │   ├── reranker.py
│   │   └── query_transform.py
│   ├── llm/
│   │   ├── base.py
│   │   ├── mock_llm.py
│   │   ├── openai_llm.py
│   │   └── local_llm.py
│   ├── safety/
│   │   ├── medical_guardrails.py
│   │   ├── emergency_detector.py
│   │   ├── pii_redaction.py
│   │   └── policy.py
│   ├── evaluation/
│   │   ├── retrieval_eval.py
│   │   ├── answer_eval.py
│   │   ├── safety_eval.py
│   │   └── eval_dataset.json
│   ├── jobs/
│   │   ├── scheduler.py
│   │   ├── crawler.py
│   │   └── incremental_update.py
│   └── utils/
│       ├── logging.py
│       ├── file_utils.py
│       └── hashing.py
├── frontend/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sample/
│   └── indexes/
├── tests/
│   ├── test_chunker.py
│   ├── test_safety.py
│   ├── test_retrieval.py
│   └── test_api.py
├── scripts/
│   ├── ingest_sample_data.py
│   ├── run_eval.py
│   └── reset_demo_data.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt hoặc pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── AGENTS.md

Các tính năng bắt buộc trong MVP:

Phase 1 — Text Medical RAG:
- Cho phép ingest file .txt, .md, .pdf, .docx, .html.
- Extract text.
- Chunk theo heading/page/section nếu có thể.
- Mỗi chunk phải có metadata:
  - doc_id
  - source_path
  - source_type
  - title
  - page_number nếu có
  - section_title nếu có
  - created_at
  - hash
- Tạo embedding.
- Lưu vào Qdrant hoặc local vector store fallback.
- API chat nhận question và rag_mode.
- Trả lời có sources/citations.
- Nếu không tìm thấy context phù hợp, phải nói không đủ dữ liệu, không được bịa.

Phase 2 — Hybrid Search:
- Implement dense vector search + BM25 keyword search.
- Có hàm combine bằng Reciprocal Rank Fusion hoặc scoring đơn giản.
- Cho phép chọn:
  - naive
  - hybrid
- Trả về top_k chunks có score và metadata.

Phase 3 — Reranking:
- Tạo reranker interface.
- Có mock reranker hoặc simple lexical reranker.
- Chuẩn bị chỗ để thay bằng cross-encoder sau.
- Pipeline:
  query → retrieve top 20 → rerank top 5 → generate answer.

Phase 4 — Parent-child RAG:
- Lưu parent documents/sections và child chunks.
- Search trên child chunk.
- Khi generate, có thể lấy parent context lớn hơn.
- Dùng cho guideline hoặc PDF dài.

Phase 5 — Multi-query / RAG-Fusion:
- Tạo query_transform.py.
- Với câu hỏi phức tạp, sinh 3-5 subqueries.
- Retrieve từng subquery.
- Gộp kết quả bằng RRF.
- Trả lời tổng hợp.

Phase 6 — Image RAG:
- Upload ảnh .png/.jpg/.jpeg.
- OCR text trong ảnh.
- Tạo image caption bằng mock captioner trước.
- Lưu OCR text + caption như một document.
- Metadata phải có image_path.
- Cho phép hỏi:
  “Ảnh này nói về nội dung gì?”
  “Trong ảnh có chỉ số y tế nào?”
  “Hãy giải thích infographic này ở mức học tập.”
- Không chẩn đoán bệnh từ ảnh thật.

Phase 7 — Speech RAG:
- API nhận audio question hoặc audio lecture.
- Với audio question:
  audio → STT → text question → RAG → answer.
- Với audio lecture:
  audio → transcript → chunk → embedding → searchable.
- Có mock STT nếu chưa có provider.
- Frontend có phần upload audio file.

Phase 8 — Video RAG:
- Thiết kế module video_processor.py.
- Tách audio từ video bằng moviepy/ffmpeg nếu khả thi.
- Transcript audio.
- Optional: lấy keyframes sau này.
- Với MVP, chỉ cần transcript video trước.

Phase 9 — Agentic Router:
- Tạo agentic_router.py.
- Router quyết định:
  - Nếu câu hỏi hỏi định nghĩa/tài liệu → text RAG.
  - Nếu câu hỏi có từ “ảnh”, “hình”, “infographic”, “xét nghiệm” → image RAG.
  - Nếu câu hỏi cần so sánh nhiều nguồn → multi-query/hybrid.
  - Nếu câu hỏi cấp cứu → emergency safety response.
  - Nếu context không đủ → corrective retrieval hoặc trả lời không đủ dữ liệu.
- Router ban đầu rule-based, sau này có thể đổi sang LLM router.

Phase 10 — Corrective RAG:
- Sau retrieval, kiểm tra:
  - Có context không?
  - Context có liên quan không?
  - Có đủ nguồn không?
  - Có rủi ro y tế không?
- Nếu context yếu:
  - rewrite query
  - retrieve lại
  - nếu vẫn yếu thì trả lời “không đủ dữ liệu trong tài liệu đã cung cấp”.

Phase 11 — Knowledge Update:
- Tạo jobs/crawler.py và jobs/incremental_update.py.
- Cho phép nhập danh sách URL public.
- Crawl HTML cơ bản.
- Hash nội dung.
- Nếu nội dung mới/thay đổi thì re-ingest.
- Scheduler có thể chạy thủ công trước, sau đó hỗ trợ APScheduler/cron.

Phase 12 — Evaluation:
- Tạo eval_dataset.json gồm các câu hỏi mẫu:
  - câu hỏi có answer trong tài liệu
  - câu hỏi không có answer
  - câu hỏi nguy hiểm/cấp cứu
  - câu hỏi yêu cầu chẩn đoán/kê đơn
  - câu hỏi cần so sánh
- retrieval_eval.py đo:
  - hit@k
  - MRR nếu có ground truth chunk_id
- answer_eval.py kiểm tra:
  - answer có citation không
  - answer có nói không đủ dữ liệu khi cần không
  - answer có disclaimer y tế không
- safety_eval.py kiểm tra:
  - không chẩn đoán
  - không kê đơn
  - phát hiện emergency
  - khuyến nghị gặp bác sĩ/cấp cứu khi cần.

Frontend Streamlit:
- Trang upload tài liệu.
- Trang ingest dữ liệu mẫu.
- Trang chat.
- Cho phép chọn RAG mode:
  - naive
  - hybrid
  - rerank
  - parent_child
  - multi_query
  - corrective
  - agentic
- Hiển thị:
  - answer
  - sources
  - retrieved chunks
  - scores
  - mode đã dùng
  - safety warning nếu có.
- Có upload ảnh.
- Có upload audio.
- Giao diện đơn giản, sạch, dễ demo.

Backend API:
- GET /health
- POST /ingest/file
- POST /ingest/url
- POST /chat
- POST /voice/transcribe
- POST /voice/chat
- POST /image/ingest
- POST /eval/run

Yêu cầu về prompt trả lời của chatbot:
- Luôn trả lời bằng tiếng Việt mặc định.
- Cấu trúc câu trả lời:
  1. Câu trả lời ngắn gọn.
  2. Giải thích chi tiết.
  3. Nguồn đã dùng.
  4. Cảnh báo y tế nếu phù hợp.
- Không bịa nguồn.
- Nếu không có nguồn, nói rõ: “Tôi chưa tìm thấy thông tin này trong tài liệu đã được cung cấp.”
- Không dùng ngôn ngữ chắc chắn quá mức trong y tế.
- Không nói “bạn bị bệnh X”.
- Nói “thông tin trong tài liệu gợi ý rằng...” hoặc “ở mức học tập, có thể hiểu là...”

Yêu cầu bảo mật:
- Không commit .env.
- Không lưu API key.
- Không lưu dữ liệu bệnh nhân thật.
- Có pii_redaction.py để che email, số điện thoại, mã bệnh nhân giả định.
- File upload phải giới hạn extension.
- Có kiểm tra size file cơ bản.
- Có logging nhưng không log nội dung nhạy cảm.

Yêu cầu README:
README phải có:
1. Giới thiệu dự án.
2. Dự án giải quyết vấn đề gì.
3. Phạm vi an toàn y tế.
4. Kiến trúc hệ thống.
5. Các phương pháp RAG đã implement.
6. Cách cài đặt local.
7. Cách chạy bằng Docker.
8. Cách ingest dữ liệu mẫu.
9. Cách dùng chatbot.
10. Cách chạy evaluation.
11. Roadmap.
12. Demo script.
13. Portfolio highlights.
14. Hạn chế hiện tại.
15. Không dùng cho chẩn đoán/kê đơn.

Yêu cầu AGENTS.md:
Tạo file AGENTS.md để hướng dẫn các coding agent sau này:
- Không phá kiến trúc module.
- Mỗi thay đổi phải chạy test liên quan.
- Ưu tiên code rõ ràng hơn code phức tạp.
- Khi thêm provider mới, phải implement qua interface.
- Khi thêm RAG mode mới, phải thêm test và cập nhật README.
- Không đưa dữ liệu nhạy cảm vào repo.
- Không biến bot thành hệ thống chẩn đoán/kê đơn.

Yêu cầu test tối thiểu:
- test chunker hoạt động.
- test emergency detector nhận ra câu hỏi nguy hiểm.
- test guardrail chặn yêu cầu kê đơn/chẩn đoán.
- test retrieval trả về chunks.
- test chat API trả về answer + sources.
- test “không đủ dữ liệu” khi context rỗng.

Yêu cầu triển khai:
- Dockerfile cho backend.
- docker-compose.yml gồm:
  - backend
  - qdrant
  - frontend streamlit nếu tiện
- .env.example gồm:
  - LLM_PROVIDER
  - OPENAI_API_KEY
  - EMBEDDING_PROVIDER
  - QDRANT_URL
  - DATABASE_URL
  - USE_MOCK_LLM
  - USE_LOCAL_EMBEDDINGS

Chiến lược làm việc:
Bước 1: Inspect repo hiện tại và báo cáo repo đang có gì.
Bước 2: Đề xuất kế hoạch implement theo phase.
Bước 3: Implement Phase 1 trước, đảm bảo chạy được.
Bước 4: Tạo dữ liệu mẫu trong data/sample, ví dụ vài đoạn markdown giả lập về:
  - tăng huyết áp
  - đột quỵ
  - chỉ số xét nghiệm máu
  - cảnh báo cấp cứu
  - chống chỉ định thuốc ở mức học tập
Bước 5: Viết tests cho Phase 1.
Bước 6: Cập nhật README.
Bước 7: Sau khi Phase 1 chạy ổn, mới làm tiếp Phase 2, Phase 3...

Không được làm:
- Không làm tất cả phase một cách nửa vời.
- Không bỏ qua test.
- Không tạo code giả không chạy được.
- Không hard-code đường dẫn tuyệt đối.
- Không dùng dữ liệu bệnh nhân thật.
- Không đưa ra lời khuyên y tế cá nhân.
- Không bỏ qua citation trong câu trả lời.

Kết quả mong muốn ngay trong lần triển khai đầu tiên:
- Repo có cấu trúc chuẩn.
- Backend FastAPI chạy được.
- Streamlit frontend chat được.
- Ingest được data/sample.
- Chatbot trả lời bằng RAG naive.
- Có source/citation.
- Có medical safety guardrail.
- Có README và .env.example.
- Có ít nhất 5 test pass.

Hãy bắt đầu bằng cách inspect repo. Nếu repo trống, hãy scaffold dự án theo kiến trúc trên. Sau khi code xong, hãy chạy test hoặc ít nhất chỉ rõ lệnh test cần chạy và lỗi nếu có.