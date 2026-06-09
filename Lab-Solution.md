# Lab Solution Day09 - Multi-Agent, MCP, A2A

## 1. Stage 1 - Direct LLM Calling

### LLM được khởi tạo như thế nào?

LLM được tạo trong `common/llm.py` bằng hàm `get_llm()`. Hàm này trả về `ChatOpenAI` nhưng trỏ endpoint về OpenRouter:

```python
ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3,
)
```

Đã thêm `temperature=0.3` để output ổn định hơn.

### Message gửi đến LLM có cấu trúc gì?

Trong `stages/stage_1_direct_llm/main.py`, message là list gồm:

- `SystemMessage`: định nghĩa vai trò của model là chuyên gia pháp lý.
- `HumanMessage`: chứa câu hỏi của người dùng.

### Vì sao cần SystemMessage và HumanMessage?

- `SystemMessage` đặt luật chơi, domain, tone và giới hạn câu trả lời.
- `HumanMessage` là input thực tế cần model xử lý.

## 2. Stage 2 - LLM + RAG / Tools

Đã hoàn thành `exercises/exercise_2_tools.py`.

### Việc đã làm

- Thêm entry `labor_law` vào `LEGAL_KNOWLEDGE`.
- Tạo tool `check_statute_of_limitations(case_type: str)`.
- Thêm tool mới vào danh sách `tools`.
- Thêm xử lý tool call cho `check_statute_of_limitations`.
- Cập nhật thêm bản demo `stages/stage_2_rag_tools/main.py` với cùng knowledge/tool.

### Lệnh chạy

```powershell
uv run python exercises/exercise_2_tools.py
```

## 3. Stage 3 - Single ReAct Agent

Đã cập nhật `stages/stage_3_single_agent/main.py`.

### Việc đã làm

- Thêm tool `search_case_law(keywords: str)`.
- Đưa tool này vào `TOOLS`.
- Tool hỗ trợ tra cứu nhanh các án lệ mẫu:
  - `Hadley v. Baxendale`
  - `Donoghue v. Stevenson`
  - `Carlill v. Carbolic Smoke Ball Co`

File Stage 3 đã stream từng bước `THINK + ACT`, `OBSERVE`, `FINAL ANSWER`, nên có thể dùng output này để debug reasoning của ReAct agent.

### Lệnh chạy

```powershell
uv run python stages/stage_3_single_agent/main.py
```

## 4. Stage 4 - Multi-Agent In-Process

Đã hoàn thành `exercises/exercise_4_multiagent.py`.

### Việc đã làm

- Implement `privacy_agent`.
- Thêm conditional routing cho keyword:
  - `data`
  - `privacy`
  - `gdpr`
  - `dữ liệu`
- Thêm `privacy_agent` vào graph.
- Thêm edge từ `privacy_agent` đến `aggregate_results`.
- Thêm `privacy_analysis` vào phần tổng hợp kết quả.

### Lệnh chạy

```powershell
uv run python exercises/exercise_4_multiagent.py
```

## 5. Stage 5 - Distributed A2A System

Stage 5 là hệ thống distributed gồm:

| Service | Port | Vai trò |
|---|---:|---|
| Registry | 10000 | Service discovery |
| Customer Agent | 10100 | Entry point nhận câu hỏi |
| Law Agent | 10101 | Orchestrator |
| Tax Agent | 10102 | Worker chuyên thuế |
| Compliance Agent | 10103 | Worker chuyên compliance |

### Flow xử lý

```text
User question
  -> Customer Agent
  -> Registry discover("legal_question")
  -> Law Agent
  -> Tax Agent + Compliance Agent chạy song song nếu cần
  -> Law Agent aggregate
  -> Customer Agent trả kết quả
```

### Lệnh chạy

```powershell
uv run python -m registry
uv run python -m customer_agent
uv run python -m law_agent
uv run python -m tax_agent
uv run python -m compliance_agent
uv run python test_client.py
```

Hoặc trên môi trường bash/macOS/Linux:

```bash
./start_all.sh
uv run python test_client.py
```

## 6. Assignment - Improve Agent Day08 bằng Supervisor-Workers

Đã tạo folder:

```text
Lab_Assignment/
```

### Kiến trúc

```text
User question
  -> SupervisorAgent
      -> SemanticSearchWorker
      -> BM25SearchWorker
      -> RerankWorker
      -> AnswerGeneratorWorker
```

### Workers

| Worker | Nhiệm vụ |
|---|---|
| `SemanticSearchWorker` | Tìm kiếm dense-style trên corpus Day08 |
| `BM25SearchWorker` | Tìm kiếm keyword/BM25 |
| `RerankWorker` | Gộp kết quả, loại trùng, rerank |
| `AnswerGeneratorWorker` | Tạo câu trả lời có citation |

Supervisor chạy hai retrieval worker song song, sau đó gọi rerank và answer worker. Đây là bản cải tiến từ pipeline tuần tự của Day08 sang pattern Supervisor-Workers.

### Lệnh chạy

```powershell
python Lab_Assignment/main.py
python Lab_Assignment/main.py "Hình phạt tàng trữ ma túy là gì?" --top-k 5
```

## 7. Ghi chú môi trường

Repo hiện chưa có `.env`, vì vậy các file gọi LLM thật cần tạo `.env` từ `.env.example` và điền `OPENROUTER_API_KEY`.

Phần `Lab_Assignment` chạy local, không cần OpenRouter API key.

