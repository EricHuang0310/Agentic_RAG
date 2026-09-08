"""集中管理所有外部端點、閾值與預算設定。

閾值的重要說明
--------------
Reranker 分數不是機率，`SCORE_ANSWERABLE` / `SCORE_FLOOR` 這類絕對門檻
只對「同一個 reranker 模型 + 同一批語料」有意義。換模型或換知識庫後
必須重新校準，作法是收一批標註資料（應回答 / 應反問 / 應拒答），
觀察三類 query 的 top1 分數分布後再決定切點。

在校準完成前，這裡的預設值沿用原本 `RERANK_SCORE_THRESHOLD = 1`，
所以行為的起點與舊版一致。
"""

import os


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


# ==================== 外部服務 ====================
RETRIEVE_URL = os.getenv(
    "ARKKB_RETRIEVE_URL",
    "http://10.13.130.13:30180/arkkb/datacenter/v1/retrieve:rerank",
)
LLM_URL = os.getenv(
    "ARKKB_LLM_URL",
    "http://10.13.60.113:30380/router/v1/chat/completions",
)

TARGET_KB_LIST = [
    kb.strip()
    for kb in os.getenv("ARKKB_KB_LIST", "osd_test_parent_phase3").split(",")
    if kb.strip()
]

RERANKER_MODEL = os.getenv("ARKKB_RERANKER_MODEL", "rerank-01")
LLM_MODEL = os.getenv("ARKKB_LLM_MODEL", "fedgpt-medium")

RETRIEVE_TIMEOUT = _f("ARKKB_RETRIEVE_TIMEOUT", 60)
LLM_TIMEOUT = _f("ARKKB_LLM_TIMEOUT", 180)

# ==================== 檢索參數 ====================
TOP_N = _i("ARKKB_TOP_N", 6)
TOP_K = _i("ARKKB_TOP_K", 66)

# ==================== 分數閾值（需校準）====================
# 這些絕對門檻是否已針對「目前這個 reranker + 目前這批語料」校準過。
#
# 預設 False，代表尚未校準。此時**不允許單憑絕對分數判定拒答**：
#   - 停用 SCORE_FLOOR 的硬否決
#   - grader 說證據足夠時直接採信，不再要求 top1 >= SCORE_ANSWERABLE
# 因為未校準的絕對門檻只是猜測，猜錯的代價是把答得出來的問題判成拒答。
# 未校準時的把關改由 LLM grader 負責（它至少讀得懂內容）。
#
# 跑過 scripts/probe_scores.py 確認分數尺度、把下面幾個門檻調成實際數值後，
# 再設 ARKKB_THRESHOLDS_CALIBRATED=1 打開完整的訊號否決機制。
THRESHOLDS_CALIBRATED = os.getenv("ARKKB_THRESHOLDS_CALIBRATED", "0") in ("1", "true", "True")

# top1 >= SCORE_ANSWERABLE 且 grader 認為足夠 -> 直接回答
SCORE_ANSWERABLE = _f("ARKKB_SCORE_ANSWERABLE", 1.0)
# top1 < SCORE_FLOOR -> 語料庫幾乎確定沒有這個主題，反問也沒用
SCORE_FLOOR = _f("ARKKB_SCORE_FLOOR", -1.0)
# top1 - top2 小於此值代表候選互相競爭（歧異訊號之一）
GAP_AMBIGUOUS = _f("ARKKB_GAP_AMBIGUOUS", 0.5)
# 判定「多分支互斥」時，至少要有幾個 chunk 具備一定證據強度
MULTI_BRANCH_MIN_STRONG = _i("ARKKB_MULTI_BRANCH_MIN_STRONG", 2)
# 具備一定證據強度的下限（比 SCORE_ANSWERABLE 寬鬆）
SCORE_SUPPORTIVE = _f("ARKKB_SCORE_SUPPORTIVE", 0.0)

# ==================== 分支處理策略 ====================
# 分支數 <= 此值且內容夠短 -> 分情境全部列出，不反問
INLINE_BRANCH_MAX = _i("ARKKB_INLINE_BRANCH_MAX", 3)
# 分支支撐內容總長度上限（字元）。超過就算分支數少也改用反問
INLINE_BRANCH_MAX_CHARS = _i("ARKKB_INLINE_BRANCH_MAX_CHARS", 3000)
# 同一個 session 最多反問幾次
MAX_CLARIFICATIONS = _i("ARKKB_MAX_CLARIFICATIONS", 2)

# ==================== 預算控制 ====================
# agent 迴圈最多跑幾輪 retrieve -> grade -> diagnose
MAX_ITERATIONS = _i("ARKKB_MAX_ITERATIONS", 3)
MAX_RETRIEVE_CALLS = _i("ARKKB_MAX_RETRIEVE_CALLS", 8)
MAX_LLM_CALLS = _i("ARKKB_MAX_LLM_CALLS", 10)
# 單次請求的總時間預算（秒）。超過就走 fallback，不再開新一輪
DEADLINE_SECONDS = _f("ARKKB_DEADLINE_SECONDS", 90)

# ==================== Session ====================
SESSION_TTL_SECONDS = _f("ARKKB_SESSION_TTL_SECONDS", 1800)

# ==================== 其他 ====================
# RRF 融合常數
RRF_K = _i("ARKKB_RRF_K", 60)
# 回應是否附帶完整 trace（正式環境可關掉，但強烈建議至少寫進 log）
INCLUDE_TRACE = os.getenv("ARKKB_INCLUDE_TRACE", "1") not in ("0", "false", "False")

# 預算耗盡且訊號顯示問題模糊時的最後手段反問（沒有預算做分支抽取）
GENERIC_CLARIFICATION = (
    "您的問題在作業規範中可能對應到多種情境，目前無法確定是哪一種。"
    "請補充說明您要辦理的業務類別、客戶身分或辦理通路，以便提供正確的作業流程。"
)

RETRIEVE_FAILED_MESSAGE = (
    "知識庫檢索服務目前無法連線，請稍後再試。為避免提供錯誤的作業流程，這裡不做推測。"
)

REFUSAL_MESSAGE = (
    "目前的知識庫中沒有查到可以回答這個問題的作業規範。"
    "為避免提供錯誤的作業流程，這裡不做推測，建議改洽詢主管單位或作業規範的維護窗口。"
)
