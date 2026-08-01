# MedTutor RAG - Tai lieu trinh bay ky thuat khi phong van

## 1. Du an giai quyet bai toan gi?

MedTutor la tro ly hoc tap y khoa co kha nang doc PDF, DOCX, HTML, Markdown,
van ban, transcript audio/video va noi dung OCR tu anh. He thong tra loi cau hoi
dua tren tai lieu da nap, kem nguon va vi tri trang/section khi co.

Day la he thong ho tro hoc tap, khong phai he thong chan doan hay ke don. Cau
hoi co dau hieu cap cuu duoc xu ly boi safety guardrail truoc retrieval. Neu
khong tim thay ngu canh du lien quan, he thong phai bao tai lieu khong du thong
tin thay vi dua mot chunk gan nhat nhung sai chu de cho LLM.

## 2. Kien truc tong the

```text
PDF/DOCX/HTML/Text/Image/Audio/Video
        |
        v
Loader/OCR/STT adapter
        |
        v
Document + metadata
        |
        v
Sentence-aware chunking
        |
        +--> SQLite metadata/parent sections
        +--> BGE-M3 embedding --> SQLite Dense hoac Qdrant
        +--> BM25 sparse index

Question
  |
  +--> PII redaction
  +--> Emergency/medical safety guardrail
  +--> Dense retrieval + BM25
  +--> Weighted Reciprocal Rank Fusion
  +--> Lexical hoac cross-encoder reranking
  +--> Diversity selection
  +--> Relevance gate
  +--> Qwen3 generation qua Ollama
  +--> Citations + medical disclaimer
```

Backend dung FastAPI. Metadata va parent section duoc luu trong SQLite. Dense
vector co hai adapter: `sqlite-dense` de chay local khong can Docker va Qdrant
cho moi truong co tinh dong thoi/production. Moi provider deu di qua interface
hoac factory, khong duoc goi truc tiep trong API route.

## 3. Du an dung nhung mo hinh nao?

### Embedding: BAAI/bge-m3

BGE-M3 duoc dung de ma hoa cau hoi va chunk thanh vector dense 1024 chieu.
Model ho tro nhieu ngon ngu, phu hop khi cau hoi bang tieng Viet nhung giao
trinh goc bang tieng Anh.

Ve kien truc, BGE-M3 thuoc ho encoder Transformer. Dau vao duoc tokenizer chia
thanh token, qua cac khoi self-attention va feed-forward de tao contextual
representations. Sentence Transformers lay bieu dien cho toan sequence, sau do
chuan hoa L2. Do cosine similarity giua query vector va document vector vi vay
co the tinh bang tich vo huong tren cac vector da chuan hoa.

Trong du an nay, BGE-M3 dang duoc su dung cho dense single-vector retrieval.
Khong nen tuyen bo da dung day du ba kha nang dense, sparse va multi-vector cua
paper BGE-M3 neu pipeline hien tai chi luu mot vector 1024 chieu cho moi chunk.

### Generator: Qwen3 4B Instruct 2507 qua Ollama

Runtime mac dinh dung `qwen3:4b-instruct-2507-q4_K_M`, la bien the non-thinking
phu hop hon cho grounded RAG so voi tag Qwen3 hybrid-thinking cu. Qwen3 4B la
causal decoder-only Transformer. Model sinh token tiep theo dua tren prompt gom
system policy, cau hoi va cac chunk da truy hoi. Ollama dong vai tro local
inference server; adapter `OllamaLLM` chiu trach nhiem tao request, timeout va
kiem tra response.

Prompt bat model:

- chi dung ngu canh duoc cung cap;
- tra loi tieng Viet;
- gan citation `[1]`, `[2]` vao nhan dinh;
- khong chan doan, ke don hay dua quyet dinh dieu tri ca nhan;
- noi ro khi ngu canh khong du.

### Reranker

Che do mac dinh dung lexical multi-signal reranker de chay xac dinh va it ton
RAM. Adapter `CrossEncoderReranker` ho tro
`BAAI/bge-reranker-v2-m3`. Cross-encoder nhan dong thoi cap
`[query, chunk]`, cho self-attention truc tiep giua token cua hai phia va xuat
mot relevance logit. Cach nay thuong xep hang chinh xac hon bi-encoder, nhung
cham hon nen chi chay tren top 20 candidates.

## 4. PDF duoc chuyen thanh gi?

PDF khong can chuyen thanh Word hay CSV. Pipeline xu ly theo cac buoc:

1. Loader doc PDF va extract text theo tung trang.
2. Moi trang duoc chuyen thanh document text co page marker va metadata.
3. Chunker tach text theo heading, paragraph va cau.
4. Moi chunk duoc luu duoi dang du lieu co cau truc:

```json
{
  "chunk_id": "pdf-...-chunk-...",
  "text": "noi dung chunk",
  "metadata": {
    "doc_id": "pdf-...",
    "source_path": "data/raw/book.pdf",
    "source_type": "pdf",
    "title": "Book title",
    "page_number": 42,
    "section_title": "Enzyme kinetics",
    "chunk_index": 120,
    "char_start": 1800,
    "char_end": 2674,
    "hash": "..."
  }
}
```

Text va metadata duoc luu trong SQLite. Chuoi
`title + section_title + chunk_text` duoc embedding thanh vector. Neu PDF scan
khong co text layer, can them OCR provider; khong nen coi ket qua extract rong
la mot tai lieu da ingest thanh cong.

## 5. Chunking nang cao

Thong so runtime hien tai:

```env
CHUNK_SIZE=900
CHUNK_OVERLAP=120
CHUNK_MIN_SIZE=180
```

Day la kich thuoc theo ky tu, khong phai token. Chunker:

- ton trong ranh gioi trang va Markdown heading;
- uu tien cat o paragraph hoac ket thuc cau;
- tao overlap theo bien text de tranh bat dau giua tu;
- ghep tiny tail vao chunk truoc neu co the;
- luu `char_start`, `char_end` va `chunk_index`;
- embedding them title va section de tang context khi chunk bat dau giua y.

Overlap giup thong tin nam sat bien khong bi mat, nhung overlap qua lon tao
duplicate candidates va tang chi phi embedding. Chunk qua nho thieu ngu canh;
chunk qua lon lam giam do chinh xac retrieval va dua nhieu noi dung thua vao
LLM. Gia tri 900/120 la baseline, can toi uu bang evaluation tren bo cau hoi
thuc te.

## 6. Retrieval va hybrid search

### Dense retrieval

Query duoc BGE-M3 ma hoa thanh vector 1024 chieu. Vector store tinh cosine
similarity va lay top candidates. Dense search manh voi paraphrase, dong nghia
va truy van Viet-Anh.

### BM25

BM25 tinh diem dua tren tan suat term, do hiem cua term trong corpus va do dai
document. Sparse search manh voi tu khoa chinh xac nhu ten enzyme, thuoc, gene,
chi so xet nghiem hoac cum tu viet tat.

### Weighted Reciprocal Rank Fusion

Hai danh sach duoc hop nhat bang:

```text
RRF(d) = sum_i weight_i / (k + rank_i(d))
```

RRF khong yeu cau score dense va BM25 cung thang do. Moi result giu lai
`dense_rank`, `dense_score`, `sparse_rank`, `sparse_score` de debug. Tang dense
weight khi cau hoi thuong la da ngon ngu/paraphrase; tang sparse weight khi
domain co nhieu keyword chinh xac.

## 7. Reranking va diversity

Pipeline rerank:

```text
retrieve top 20
  -> lexical hoac cross-encoder score
  -> sap xep lai
  -> MMR-style diversity selection
  -> top 5
```

Lexical reranker ket hop retrieval score, semantic score, reciprocal dense
rank, query coverage, title/section coverage va phrase match. Khi khong co
dense signal, no giu cong thuc lexical thuần de fallback van chay duoc.
Cross-encoder thay cac signal heuristic bang semantic score hoc duoc.

Diversity selection phat cac chunk co Jaccard similarity cao voi chunk da
chon. Tham so `RERANKER_DIVERSITY_LAMBDA=0.75` uu tien relevance 75% va
diversity 25%, giup tranh top 5 deu la cac doan overlap gan nhu giong nhau.

## 8. Cac RAG mode

- `naive`: dense/local retrieval truc tiep.
- `hybrid`: dense + BM25 + weighted RRF.
- `rerank`: hybrid candidates + reranker + diversity.
- `parent_child`: tim child chunk, mo rong parent section khi generate.
- `multi_query`: tao nhieu bien the query, retrieve tung query, gop bang RRF.
- `corrective`: kiem tra relevance, rewrite va retrieve lai mot lan.
- `agentic`: rule-based router chon pipeline theo loai cau hoi.

Guardrail cap cuu chay truoc tat ca mode. Relevance gate chay truoc generation
va chap nhan result khi co topic anchor lexical hoac semantic score vuot
threshold. Khong co result dat gate thi he thong tra ve thong bao thieu du lieu
va khong tao citation.

## 9. Do chinh xac duoc danh gia nhu the nao?

Khong co mot con so "accuracy cua RAG" duy nhat. Du an tach thanh:

- `Hit@k`: nguon dung co nam trong top-k khong.
- `MRR`: nguon dung duoc xep o vi tri cao den dau.
- citation rate: cau tra loi grounded co citation hay khong.
- format compliance: answer co du bon phan bat buoc hay khong.
- insufficient-context accuracy: cau ngoai corpus co bi tu choi dung khong.
- disclaimer rate: cau tra loi y khoa co disclaimer bat buoc khong.
- safety accuracy: emergency va yeu cau chan doan/ke don co bi xu ly dung khong.

Bo eval hien tai la bo smoke test nho, khong phai clinical benchmark va khong
du de cong bo model dat do chinh xac y khoa. Khi phong van, can noi ro ket qua
do chat luong cua pipeline tren bo cau hoi noi bo; no khong chung minh tinh
dung dan lam sang.

Snapshot local ngay 2026-07-31 tren 6 eval cases:

```text
Hit@5:                         1.000
MRR:                           0.833
Citation rate:                 1.000
Format compliance rate:        1.000
Disclaimer rate:               1.000
Insufficient-context accuracy: 1.000
Safety accuracy:               1.000
```

Ket qua nay dung BGE-M3, SQLite dense + BM25, Qwen3 4B Instruct 2507 va
lexical-semantic reranker. Day la pipeline regression snapshot, khong phai
clinical accuracy.

De danh gia nghiem tuc hon, can:

1. Tao bo 100-500 cau hoi tu chinh corpus.
2. Gan nhan relevant document/chunk boi nguoi co kien thuc domain.
3. Chia theo exact keyword, paraphrase, Viet-Anh, multi-hop va unanswerable.
4. So sanh BM25, dense, hybrid, hybrid + reranker theo cung mot bo cau hoi.
5. Theo doi Recall@5/10, nDCG@10, MRR, faithfulness va citation correctness.
6. Kiem tra regression moi khi doi chunk size, model hoac fusion weight.

## 10. Vi sao he thong tung tra ve doan khong lien quan?

Vector search luon co "nearest neighbor", ke ca khi tat ca neighbors deu sai.
Nhung nguyen nhan thuong gap:

- dung lexical embedding cho cau hoi Viet-Anh;
- vector index chua duoc rebuild sau khi ingest/chunk lai;
- chunk mat title/section nen thieu context;
- chunk size hoac overlap khong phu hop;
- top-k qua nho;
- BM25 va dense weight khong phu hop;
- khong co relevance threshold/reranker;
- corpus tron nhieu tai lieu khac chu de.

Du an xu ly bang contextual indexing, BGE-M3 multilingual embedding, hybrid
retrieval, reranking, diversity, relevance gate va clean re-ingest. Endpoint
`GET /retrieve` tra cac signal de xem result den tu dense hay sparse va tai sao
no duoc xep hang.

## 11. Lenh chay

Backend voi cau hinh `.env`:

```powershell
D:\App\Anaconda\Miniconda\python.exe -m uvicorn app.main:app --env-file .env --port 8000
```

Rebuild toan bo dense index:

```powershell
D:\App\Anaconda\Miniconda\python.exe -m scripts.reindex_vectors --clear --batch-size 32
```

Tiep tuc neu bi gian do:

```powershell
D:\App\Anaconda\Miniconda\python.exe -m scripts.reindex_vectors --resume --batch-size 32
```

Frontend:

```powershell
cd web
npm run dev
```

Kiem tra:

```powershell
D:\App\Anaconda\Miniconda\python.exe -m pytest -q
D:\App\Anaconda\Miniconda\python.exe -m scripts.run_eval
```

## 12. Gioi han va huong nang cap

- `sqlite-dense` quet vector trong process, phu hop local corpus nho/vua;
  production nen dung Qdrant va tao payload index.
- Lexical reranker la baseline; cross-encoder can them RAM va latency.
- PDF scan can OCR that; mock image caption/STT khong dai dien chat luong
  production.
- Evaluation dataset can duoc mo rong va gan nhan boi domain expert.
- Can them observability cho latency tung stage, cache query embedding,
  batching va version hoa index theo model/chunk config.
- Can kiem thu prompt injection trong tai lieu va phan quyen theo collection
  truoc khi dung voi tai lieu noi bo.
