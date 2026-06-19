# Learning MVP: RAG từ số 0

Thư mục này là bản học tập tối giản. Mục tiêu là hiểu luồng RAG trước khi dùng FastAPI, Qdrant, embedding model, OCR, audio, hoặc LLM thật.

Pipeline nhỏ này có 7 bước:

```text
file .md/.txt
  -> ingest: đọc file thành document
  -> chunk: tách document thành đoạn nhỏ
  -> store: lưu chunks vào JSON
  -> search: tìm chunk bằng keyword đơn giản
  -> chat: ghép question + chunks thành câu trả lời mock có nguồn
  -> build index: gom nhiều tài liệu thành một kho chunks
  -> tf-idf search: xếp hạng chunk bằng trọng số từ hiếm
```

## Chạy từng bước

Từ root repo:

```bash
python learning_mvp/01_ingest.py
python learning_mvp/02_chunk.py
python learning_mvp/03_store.py
python learning_mvp/04_search.py "FAST dot quy"
python learning_mvp/05_chat.py "FAST trong đột quỵ gồm những dấu hiệu nào?"
python learning_mvp/06_build_index.py
python learning_mvp/04_search.py "hemoglobin xét nghiệm" --chunks learning_mvp/index.json
python learning_mvp/05_chat.py "Tài liệu nói gì về hemoglobin?" --chunks learning_mvp/index.json
python learning_mvp/07_tfidf_search.py "hemoglobin xét nghiệm" --chunks learning_mvp/index.json
python learning_mvp/05_chat.py "Tài liệu nói gì về hemoglobin?" --chunks learning_mvp/index.json --search-mode tfidf
```

## Học gì ở từng file

- `01_ingest.py`: đọc file thành một `document`.
- `02_chunk.py`: tách document thành nhiều `chunk`.
- `03_store.py`: lưu chunk xuống `learning_mvp/chunks.json`.
- `04_search.py`: tìm chunk liên quan bằng keyword overlap.
- `05_chat.py`: tạo câu trả lời mock từ chunk tìm được và hiển thị nguồn.
- `06_build_index.py`: đọc nhiều file trong `data/sample`, chunk tất cả, lưu thành `learning_mvp/index.json`.
- `07_tfidf_search.py`: search giống bước 4 nhưng dùng TF-IDF để ưu tiên từ đặc trưng hơn.
- `core.py`: chứa các hàm nhỏ dùng chung để dễ test.

## `chunks.json` khác `index.json` thế nào?

- `chunks.json`: thường đến từ một file duy nhất, dùng để học từng bước.
- `index.json`: gom chunks của nhiều tài liệu, gần với cách RAG thật hoạt động hơn.

## Vì sao chưa dùng embedding?

Embedding sẽ dễ hiểu hơn sau khi bạn nắm chắc 3 ý:

1. Document là gì.
2. Chunk là gì.
3. Retrieval là gì.

Sau đó ta có thể thay `search_chunks()` bằng embedding search mà không đổi toàn bộ luồng.

## Keyword search khác TF-IDF thế nào?

- Keyword search hiện tại đếm số từ query xuất hiện trong chunk.
- TF-IDF vẫn là lexical search, nhưng tăng trọng số cho từ hiếm trong index.
- Ví dụ `hemoglobin` thường đặc trưng hơn các từ chung như `tài liệu`, `nói`, `gì`.
