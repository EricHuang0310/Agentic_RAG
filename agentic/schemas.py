"""API 與內部資料結構。

對外回應刻意保留舊版 v2.2.0 的所有欄位（session_id / answer /
references_summary / reference_chunks / retrieve_db / top_score / status /
router_round），新欄位都是附加的，舊前端不需改動即可運作。
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ==================== 決策型別 ====================
class Diagnosis(str, Enum):
    """檢索不足時的失敗類型診斷。"""

    SUFFICIENT = "sufficient"                # 證據足夠，直接回答
    LEXICAL_MISMATCH = "lexical_mismatch"    # 主題對但措辭對不上 -> agent 自己改寫重試
    COMPOUND = "compound"                    # 複合問題 -> 拆解後各自檢索
    MULTI_BRANCH = "multi_branch"            # 多個分支都對，取決於使用者情境
    OUT_OF_SCOPE = "out_of_scope"            # 語料庫沒有這個主題 -> 拒答
    UNKNOWN = "unknown"                      # grader 失效時的保守值


class Decision(str, Enum):
    """最終採取的行動。"""

    ANSWER = "answer"                        # 單一情境直接回答
    ANSWER_ALL_BRANCHES = "answer_all_branches"  # 分情境全部列出（省一輪往返）
    CLARIFY = "clarify"                      # 反問（帶選項）
    REFUSE = "refuse"                        # 拒答 + 轉人工


# ==================== 內部資料結構 ====================
class Chunk(BaseModel):
    """檢索到的單一 chunk。"""

    chunk_id: str = Field(..., description="本次請求內的穩定識別碼")
    filename: str
    page: str = ""
    score: float = Field(..., description="該 chunk 的 Reranker 分數（多變體檢索時取最大值）")
    content: str = ""
    rrf_score: float = Field(0.0, description="多變體檢索融合後的 RRF 分數")
    matched_variants: List[str] = Field(default_factory=list, description="命中此 chunk 的 query 變體名稱")


class Signals(BaseModel):
    """從一次檢索結果算出的數值訊號，用來仲裁 LLM grader 的判斷。"""

    top1: float = 0.0
    top2: float = 0.0
    gap: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    n_chunks: int = 0
    n_supportive: int = Field(0, description="分數 >= SCORE_SUPPORTIVE 的 chunk 數")
    n_answerable: int = Field(0, description="分數 >= SCORE_ANSWERABLE 的 chunk 數")
    n_sources: int = Field(0, description="不同來源檔案數")
    source_entropy: float = Field(0.0, description="分數權重在來源檔案間的正規化熵，越高越分散")
    max_source_share: float = Field(0.0, description="最集中來源檔案佔的分數權重比例")


class GradeResult(BaseModel):
    """LLM grader 的輸出。"""

    verdict: Diagnosis = Diagnosis.UNKNOWN
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    missing_information: str = ""
    reasoning: str = ""
    parse_ok: bool = True


class Branch(BaseModel):
    """一個互斥的作業情境分支。"""

    label: str = Field(..., description="分支名稱，例如「久未往來帳戶」")
    key_terms: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)


class TraceStep(BaseModel):
    """單一步驟的執行紀錄，是日後調閾值與除錯的唯一依據。"""

    step: str
    latency_ms: int = 0
    detail: Dict[str, Any] = Field(default_factory=dict)


# ==================== 對外 Schema ====================
class ReferenceChunk(BaseModel):
    filename: str = Field(..., description="來源檔案名稱")
    page: str = Field(..., description="頁數")
    score: float = Field(..., description="該 Chunk 的 Reranker 分數")
    content: str = Field(..., description="原始 Chunk 內文內容")


class ClarificationOption(BaseModel):
    """結構化反問選項，前端可直接渲染成按鈕。"""

    label: str = Field(..., description="顯示給使用者的選項文字")
    value: str = Field(..., description="使用者點選後要回傳的值")


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="對話 Session ID（第一輪不帶，後續請帶回上一輪回傳的 ID）")
    message: str = Field(..., description="使用者輸入訊息", examples=["遺留物現金怎麼處理？"])
    include_trace: Optional[bool] = Field(None, description="是否回傳完整 agent trace，預設依伺服器設定")


class ChatResponse(BaseModel):
    # --- 舊版欄位（保持相容）---
    session_id: str = Field(..., description="對話 Session ID")
    answer: str = Field(..., description="LLM 回答內容或反問引導語")
    references_summary: str = Field(..., description="簡短參考檔案與頁數清單（字串）")
    reference_chunks: List[ReferenceChunk] = Field(default_factory=list, description="檢索到的所有原始 Chunk 詳細內容清單")
    retrieve_db: Union[list, str] = Field(..., description="檢索使用的知識庫名稱")
    top_score: float = Field(..., description="最高 Reranker 相關分數")
    status: str = Field(..., description="對話狀態 (completed / awaiting_clarification / refused)")
    router_round: str = Field(..., description="對話階段說明")

    # --- 新增欄位 ---
    decision: Decision = Field(..., description="agent 最終採取的行動")
    diagnosis: Diagnosis = Field(..., description="檢索結果的失敗類型診斷")
    clarification_options: List[ClarificationOption] = Field(
        default_factory=list, description="反問時提供的結構化選項"
    )
    effective_query: str = Field("", description="實際送去檢索的 query（可能已改寫或重構）")
    signals: Optional[Signals] = Field(None, description="檢索結果的數值訊號")
    iterations: int = Field(0, description="agent 迴圈實際跑的輪數")
    budget_exhausted: bool = Field(False, description="是否因預算耗盡而提前收斂")
    trace: List[TraceStep] = Field(default_factory=list, description="逐步執行紀錄")
