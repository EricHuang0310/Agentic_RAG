# Agentic RAG — 反問／拒答決策服務

內部作業流程（SOP）知識庫問答服務。核心設計是把「要不要反問使用者」從
單一 reranker 分數門檻，改成一個有診斷、有重試、有預算上限的 agent 迴圈。

## 為什麼改

舊版的判斷是 `if top_score < RERANK_SCORE_THRESHOLD: 反問`，而且只在第一輪
反問，第二輪不論分數多低都會硬答。這個設計把兩種本質不同的失敗混成一種：

| 失敗類型 | 資訊在哪裡 | 正確處置 |
|---|---|---|
| 措辭對不上語料（口語 vs SOP 術語） | 在語料庫裡，只是沒撈到 | agent 自己改寫重試 |
| 語料有多個分支都對，取決於使用者情境 | **只在使用者腦袋裡** | 反問（或分情境全列） |
| 語料庫沒有這份規範 | 不存在 | 拒答並轉人工 |

第一種反問是浪費使用者一輪，第三種反問了也沒用。只有第二種該問。

判斷準則：**把 query 重寫一百次，語料庫裡會出現「唯一」答案嗎？**
不會（因為有多段都對，差別在情境）才是真的需要反問。

## 決策流程

```
retrieve → grade
  ├ 夠好 ───────────────────────────→ answer
  └ 不夠 → 診斷失敗類型
       (a) 措辭不匹配   → 改寫 / HyDE / 多變體 fan-out 重試   ← 自己解決，不問
       (b) 複合問題     → 拆解成子問題各自檢索                ← 自己解決，不問
       (c) 多分支互斥   → 分支 ≤3 且答案短 → answer_all_branches（分情境全列）
                         分支多 / 答案長  → clarify（帶結構化選項）
       (d) 全面低分分散 → refuse（拒答 + 轉人工）
  預算耗盡 → clarify 或 refuse，絕不硬答
```

反問在這裡是**最後手段**，不是 pipeline 的入口閘門。

`(c)` 那個「分支 ≤3 且答案短就直接全列」的分支是實務上最容易被忽略的省一輪
機會：與其問「您是久未往來還是新開戶？」，不如直接兩種都列出來讓使用者自己挑，
順便讓他知道有這個區別存在。

## 診斷的仲裁原則：LLM 提議、數值訊號否決

LLM grader 讀得懂內容，但對「我的證據夠不夠」的自我評估有偏（傾向說夠）；
分數訊號沒有語意但誠實。所以兩者都用，衝突時由訊號否決：

- grader 說 `sufficient` 而 top1 未達門檻 → 不接受，先去找更強的證據
- grader 說 `multi_branch` 而只有單一來源、gap 大 → 不是分支問題
- grader 說 `not_in_corpus` 而分數其實不低 → 可能只是措辭沒對上，值得改寫一次
- grader 輸出無法解析 → 退回純訊號規則，且標記 `low_confidence`

數值訊號不只有 top1（`agentic/signals.py`）：

| 訊號 | 意義 |
|---|---|
| `top1` / `top2` / `gap` | gap 小代表候選互相競爭（歧異訊號） |
| `n_supportive` / `n_answerable` | 有多少 chunk 真的具備證據強度 |
| `n_sources` / `source_entropy` / `max_source_share` | **來源分散度**，區分「模糊」與「無資料」的關鍵 |

- 高分 + 集中單一檔案 → 證據明確
- 高分 + 散在多檔 + gap 小 → 多分支互斥（該問）
- 全面低分 + 散在多檔 → 語料庫沒有（該拒答）

## 模組職責

| 模組 | 內容 |
|---|---|
| `arkkb_router_search_all_db_api.py` | FastAPI 入口（維持原檔名與 port 9454） |
| `agentic/config.py` | 所有閾值、端點、預算，全部可用環境變數覆寫 |
| `agentic/orchestrator.py` | agent 主迴圈 |
| `agentic/diagnose.py` | 失敗類型診斷（**純函式**，可離線調參） |
| `agentic/policy.py` | 多分支時「全列 vs 反問」的策略（**純函式**） |
| `agentic/signals.py` | 數值訊號（**純函式**） |
| `agentic/fusion.py` | 多變體結果解析與 RRF 融合（**純函式**） |
| `agentic/grading.py` | LLM grader、互斥分支抽取 |
| `agentic/query_ops.py` | standalone 重構、術語改寫、關鍵詞、HyDE、複合問題拆解 |
| `agentic/generate.py` | 回答生成（保留原本嚴格 prompt）+ 引用編號驗證 |
| `agentic/budget.py` | 迴圈上限、呼叫次數、時間預算 |
| `agentic/sessions.py` | 多輪狀態（TTL + 鎖） |
| `agentic/trace.py` | 逐步執行紀錄 |

診斷與策略刻意寫成不碰網路的純函式，所以閾值可以拿歷史 trace 離線重跑驗證，
不需要打真的服務。

## 執行

```bash
pip install -r requirements.txt
python arkkb_router_search_all_db_api.py     # http://0.0.0.0:9454
pytest                                        # 59 個測試，不連外部服務
```

## API

`POST /api/v1/chat`

```json
{ "session_id": null, "message": "遺留物現金怎麼處理？", "include_trace": true }
```

回應保留舊版 v2.2.0 的**所有欄位**（`session_id` / `answer` /
`references_summary` / `reference_chunks` / `retrieve_db` / `top_score` /
`status` / `router_round`），舊前端不改也能運作。新增欄位：

| 欄位 | 說明 |
|---|---|
| `decision` | `answer` / `answer_all_branches` / `clarify` / `refuse` |
| `diagnosis` | `sufficient` / `lexical_mismatch` / `compound` / `multi_branch` / `out_of_scope` |
| `clarification_options` | 結構化反問選項，前端可直接渲染成按鈕（末項為「以上皆非」） |
| `effective_query` | 實際送去檢索的 query（可能已重構／改寫） |
| `signals` | 該次檢索的完整數值訊號 |
| `iterations` / `budget_exhausted` | 迴圈輪數與是否因預算收斂 |
| `trace` | 逐步執行紀錄 |

`status` 多了 `refused`（原本只有 `completed` / `awaiting_clarification`）。

`GET /healthz` 回傳目前生效的閾值與預算設定，方便確認環境變數有沒有吃到。

## 反問話術不再寫死

舊版把四個業務類別硬編碼在 `CLARIFICATION_MESSAGE` 裡。現在反問內容由
`agentic/policy.py` 依檢索結果抽出的分支動態組出來，換知識庫或新增業務類別
不需要改任何程式碼。反問也不會重複問同一個維度（`asked_dimensions`），
次數達 `MAX_CLARIFICATIONS` 後改為分情境全列。

## 成本與延遲

Happy path（證據充足）只花 **1 次檢索 + 1 次 grading + 1 次生成**，
query 分析要等到第一輪判定不足才會做。最壞情況由預算上限夾住：

| 設定 | 預設 | 說明 |
|---|---|---|
| `ARKKB_MAX_ITERATIONS` | 3 | 最多幾輪 retrieve → grade → diagnose |
| `ARKKB_MAX_RETRIEVE_CALLS` | 8 | 含多變體 fan-out |
| `ARKKB_MAX_LLM_CALLS` | 10 | 含 grading、分析、分支抽取、生成 |
| `ARKKB_DEADLINE_SECONDS` | 90 | 總時間預算，超過就收斂到反問或拒答 |

預算耗盡時的行為是明確的：**絕不硬答**。訊號顯示有競爭證據就反問，
證據普遍不足就拒答。

## 閾值

預設值假設 reranker 輸出 **0～1 的正規化分數**：

| 設定 | 預設 | 依據 | 作用 |
|---|---|---|---|
| `ARKKB_SCORE_ANSWERABLE` | `0.7` | **實測** | 達到就直接採信 grader 的「證據足夠」 |
| `ARKKB_SCORE_SUPPORTIVE` | `0.35` | 依尺度推導 | 算得上有證據的下限；改寫後仍低分時的可答下限 |
| `ARKKB_GAP_AMBIGUOUS` | `0.08` | 依尺度推導 | top1−top2 小於此值視為候選互相競爭 |
| `ARKKB_SCORE_FLOOR` | 停用 | — | 唯一「不管 grader 說什麼都拒答」的門檻 |
| `ARKKB_THRESHOLDS_CALIBRATED` | `1` | — | 關掉則完全不看絕對分數，把關全交給 grader |

`0.7` 在新架構裡的意義和舊版**不同**：舊版是「低於 0.7 就反問」，這裡是
「低於 0.7 就先去找更強的證據」，之後才可能反問。所以反問率會比舊版低——
一部分低分問題會在改寫後直接答出來，這正是改架構的目的。

`SCORE_FLOOR` 預設停用，因為它設錯就會把答得出來的問題判成沒有資料。
等手上有「應拒答」的標註資料、確認那些問題的 top1 落在哪個上緣之後再打開
（`ARKKB_SCORE_FLOOR=0.12`，要關掉設 `off`）。停用期間拒答一律需要 grader
判定，`GET /healthz` 的 `thresholds.floor` 會是 `null`。

換 reranker 或換知識庫後這些值全部要重測。建議流程：

1. 收 50～100 題真實問題，標註「應回答 / 應反問 / 應拒答」
2. 跑 `python scripts/probe_scores.py --file questions.txt`（每行 `問題<TAB>標註`）
3. 看三類問題的 `top1` 與 `gap` 分布，依它建議的切點微調
4. 設定 `ARKKB_THRESHOLDS_CALIBRATED=1`
5. 量兩個對立指標：
   - **over-clarification rate**：本來能直接答卻反問了
   - **silent misinterpretation rate**：該反問卻自己挑一個情境答了

只看前者會讓人把反問拿掉，然後在後者上悄悄爆掉——而後者才是真的會造成
作業錯誤的那個。

## 排查：第一輪就直接拒答

回應的 `trace` 裡有一步 `diagnose`，它的 `reason` 會直接寫出是哪一條規則
判定拒答的，對照下表處理：

| `reason` | 意思 | 怎麼處理 |
|---|---|---|
| `檢索結果為空` | 知識庫回了 200 但 `data` 是空陣列 | 檢查 `kb_list` 名稱、`top_k`；用 `scripts/probe_scores.py` 直接打同一個 query 確認 |
| `top1=… 低於地板 …` | 絕對門檻否決 | 只有設了 `ARKKB_SCORE_FLOOR` 才會出現；值設太高 |
| `grader 說足夠但 top1=… 未達門檻` | 分數低於 `SCORE_ANSWERABLE` | 正常行為，會接著改寫重試；若沒看到 `analyze_query` 才是問題 |
| `grader 判定語料庫未涵蓋此主題` | LLM grader 說沒有 | 看 `grade` 那一步的 `reasoning`；若判斷有誤是 grader prompt 的問題 |
| `grader 失效；分數不足以支撐作答` | grader 輸出無法解析，退回純訊號 | 看 `grade` 的 `parse_ok`；門檻沒校準時分數判斷不可信 |

`trace` 裡的 `analyze_query` 有一個 `trigger` 欄位，說明這次分析是為什麼做的
（`grade_insufficient` = grader 判定不足；`retry_needs_variants` = 診斷判定要改寫
但先前略過了分析）。若診斷是 `lexical_mismatch` 卻完全沒有 `analyze_query`
這一步，代表改寫根本沒發生。

另外看 `signals` 那一步的 `top1`：**如果它是負數或遠小於 1，就代表預設門檻
和你的 reranker 尺度對不上**。拒答時 `reference_chunks` 仍會回傳，裡面的
`score` 也可以直接看出尺度。

先跑一次探測：

```bash
python scripts/probe_scores.py "遺留物現金怎麼處理？" "銷戶要怎麼辦理？"
```

它會印出真實的 top1／top2／gap 分布。有標註資料（每行 `問題<TAB>answer|clarify|refuse`）
時還會建議切點。把 `agentic/config.py` 的門檻改成實際數值後，設定
`ARKKB_THRESHOLDS_CALIBRATED=1` 才會啟用完整的訊號否決。

`GET /healthz` 的 `thresholds.calibrated` 可以確認目前是哪個模式。

## 排查：被問了一模一樣的問題

反問防護有三層，前兩層依賴 session 狀態、第三層不依賴：

1. `clarification_count >= MAX_CLARIFICATIONS` → 改為分情境全列
2. 同一個維度或同一組分支標籤問過 → 改為分情境全列
3. **訊息內容本身已指定分支** → 直接回答該分支（不需要 session）

若還是被重複問，先看回應 `trace` 裡的 `session_lookup` 那一步：

- `session_found: false` 而 `session_id_provided: true` → session 遺失。
  常見原因是 TTL（預設 30 分鐘）過期、`reload=True` 期間服務重啟、
  或 uvicorn 多 worker 各持一份記憶體狀態。
- `session_id_provided: false` → 前端第二輪沒有把第一輪的 `session_id` 帶回來。

再看 `router_round`：第二輪應該是「反問後補充輪」，若顯示「首輪提問」
就代表這一輪被當成全新問題處理了。

第 3 層防護要求**恰好一個**分支匹配才生效。像「變更負責人」同時命中
「獨資戶變更負責人」與「公司變更負責人」時仍屬模糊，會繼續反問——這是刻意的。

## 目前的限制

- **沒有 lexical 通道。** 知識庫 API 只吃單一 query 字串，沒有 BM25 或 hybrid
  端點，所以「hybrid」是以 multi-query fan-out + RRF 近似（原問法、術語改寫、
  關鍵詞式、HyDE 假想條文）。真正的 BM25 要等後端支援。
- **只驗證引用編號，沒有句級 grounding。** `generate.py` 會檢查
  `[文獻 N]` 的 N 是否存在並在回答尾端標註異常，但不做 NLI 或句級比對——
  那需要額外的模型呼叫，屬於下一階段。
- **session 是記憶體實作。** 已加 TTL 與鎖，但 uvicorn 多 worker 時每個 worker
  各有一份。正式環境請照 `sessions.py` 的介面（`get` / `set` / `delete`）
  換成 Redis。
- **沒有 metadata / 版本過濾。** SOP 有版次與生效日期，目前不會優先取最新版，
  也不會在新舊衝突時主動說明。
- **`TARGET_KB_LIST` 仍是固定的。** 尚未做 KB routing。
