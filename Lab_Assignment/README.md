# Lab Assignment Day09 - Improve Agent Day08

## Mục tiêu

Yêu cầu trong `Lab-assignment-checklist.md`: cải tiến Agent Day08 bằng pattern **Supervisor - Workers** với ít nhất 2-3 workers. Phần này dùng lại dữ liệu và ý tưởng RAG của Day08, nhưng tổ chức lại thành nhiều worker chuyên trách.

## Kiến trúc

```text
User question
  -> SupervisorAgent
      -> SemanticSearchWorker  (dense-style retrieval)
      -> BM25SearchWorker      (lexical retrieval)
      -> RerankWorker          (merge + rerank)
      -> AnswerGeneratorWorker (answer with citations)
```

Supervisor chạy `SemanticSearchWorker` và `BM25SearchWorker` song song, sau đó gom kết quả, loại trùng, rerank, rồi tạo câu trả lời cuối có citation.

## Workers

| Worker | Vai trò |
|---|---|
| `SemanticSearchWorker` | Tìm đoạn liên quan bằng embedding nhẹ local trên corpus Day08 |
| `BM25SearchWorker` | Tìm đoạn liên quan bằng BM25 keyword search |
| `RerankWorker` | Gộp kết quả từ các worker, ưu tiên đoạn có overlap cao và được nhiều worker đồng thuận |
| `AnswerGeneratorWorker` | Tạo câu trả lời trích xuất, có citation nguồn |

## Dữ liệu

Folder này đã copy dữ liệu cần thiết từ Day08 vào:

```text
Lab_Assignment/data/standardized/
Lab_Assignment/data/vectorstore/local_index.json
```

Nếu muốn dùng repo Day08 gốc, có thể set:

```powershell
$env:DAY08_RAG_ROOT="D:\DSA\Day08_RAG_pipeline_cohort2"
```

## Cách chạy

Từ root repo Day09:

```powershell
python Lab_Assignment/main.py
```

Hoặc truyền câu hỏi riêng:

```powershell
python Lab_Assignment/main.py "Hình phạt tàng trữ ma túy là gì?" --top-k 5
```

## Điểm cải tiến so với Day08

- Day08 ban đầu là pipeline tuần tự: semantic search + lexical search + rerank + generation.
- Day09 chuyển thành multi-agent local pattern:
  - Supervisor điều phối luồng xử lý.
  - Mỗi worker có trách nhiệm riêng, dễ thay thế hoặc mở rộng.
  - Hai retrieval worker chạy song song để giảm latency.
  - Trace cho biết worker nào chạy, số hit, thời gian và nguồn được chọn.

