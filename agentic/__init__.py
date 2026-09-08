"""Agentic RAG 元件。

模組職責：

    config       閾值、端點、預算等所有可調參數
    schemas      對外 API 與內部資料結構
    clients      知識庫與 LLM 的非同步用戶端
    budget       預算控制（迴圈上限、呼叫次數、時間）
    trace        逐步執行紀錄
    signals      從檢索結果算出的數值訊號（純函式）
    fusion       多變體結果的解析與 RRF 融合（純函式）
    query_ops    query 重構、改寫、HyDE、複合問題拆解
    grading      LLM grader 與互斥分支抽取
    diagnose     失敗類型診斷（純函式，LLM 提議 / 訊號否決）
    policy       多分支時「全列 vs 反問」的策略（純函式）
    generate     回答生成與引用驗證
    sessions     多輪對話狀態
    orchestrator agent 主迴圈
"""

__all__ = [
    "config",
    "schemas",
    "clients",
    "budget",
    "trace",
    "signals",
    "fusion",
    "query_ops",
    "grading",
    "diagnose",
    "policy",
    "generate",
    "sessions",
    "orchestrator",
]
