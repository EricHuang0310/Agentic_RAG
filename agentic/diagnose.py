"""失敗類型診斷：純函式，不呼叫任何外部服務，方便單元測試與離線調參。

仲裁原則：**LLM grader 提議，數值訊號否決**。

grader 讀得懂內容但對「自己夠不夠」的評估有偏（傾向說夠）；分數訊號
沒有語意但誠實。因此：

  - grader 說 sufficient 而 top1 低於門檻 -> 不接受，先去找更好的證據
  - grader 說 multi_branch 而只有一個 chunk 有證據 -> 不是分支問題
  - grader 說 not_in_corpus 而分數其實不低 -> 可能只是措辭沒對上，值得改寫重試
  - grader 完全失效（輸出無法解析）-> 退回純訊號規則，且偏保守
"""

from dataclasses import dataclass
from typing import Optional

from . import config
from .schemas import Diagnosis, GradeResult, Signals


@dataclass
class DiagnosisOutcome:
    diagnosis: Diagnosis
    reason: str
    low_confidence: bool = False

    def as_detail(self) -> dict:
        return {
            "diagnosis": self.diagnosis.value,
            "reason": self.reason,
            "low_confidence": self.low_confidence,
        }


def diagnose(
    signals: Signals,
    grade: GradeResult,
    *,
    retried_variants: bool,
    decomposed: bool,
    is_compound: Optional[bool] = None,
) -> DiagnosisOutcome:
    """判斷這次檢索屬於哪一種失敗類型。

    參數
    ----
    retried_variants: 是否已經做過改寫 / HyDE / 多變體 fan-out。
        沒做過時遇到疑似措辭問題應該先重試，做過了就不要再繞。
    decomposed: 是否已經拆解過複合問題。
    is_compound: query 分析的結果；None 代表還沒分析過。
    """
    # 1. 沒有任何檢索結果
    if signals.n_chunks == 0:
        return DiagnosisOutcome(Diagnosis.OUT_OF_SCOPE, "檢索結果為空")

    # 2. 硬地板否決：分數低到這種程度時，grader 說什麼都不採信
    if signals.top1 < config.SCORE_FLOOR:
        return DiagnosisOutcome(
            Diagnosis.OUT_OF_SCOPE,
            f"top1={signals.top1} 低於地板 {config.SCORE_FLOOR}，語料庫應無此主題",
        )

    # 3. grader 說足夠
    if grade.verdict == Diagnosis.SUFFICIENT:
        if signals.top1 >= config.SCORE_ANSWERABLE:
            return DiagnosisOutcome(Diagnosis.SUFFICIENT, "grader 判定足夠且分數達門檻")
        if not retried_variants:
            return DiagnosisOutcome(
                Diagnosis.LEXICAL_MISMATCH,
                f"grader 說足夠但 top1={signals.top1} 未達門檻，先改寫找更強證據",
            )
        if signals.top1 >= config.SCORE_SUPPORTIVE:
            return DiagnosisOutcome(
                Diagnosis.SUFFICIENT,
                "重試後 grader 仍判定足夠，分數偏低但可作答",
                low_confidence=True,
            )
        return DiagnosisOutcome(
            Diagnosis.OUT_OF_SCOPE,
            f"重試後 top1={signals.top1} 仍過低，不接受 grader 的 sufficient",
        )

    # 4. 複合問題優先拆解（自己能解決，不要問使用者）
    if is_compound and not decomposed:
        return DiagnosisOutcome(Diagnosis.COMPOUND, "偵測到複合問題，先拆解成子問題各自檢索")

    # 5. grader 說多分支互斥
    if grade.verdict == Diagnosis.MULTI_BRANCH:
        if has_branch_evidence(signals):
            return DiagnosisOutcome(Diagnosis.MULTI_BRANCH, "多個來源都有相當證據且彼此競爭")
        if not retried_variants:
            return DiagnosisOutcome(
                Diagnosis.LEXICAL_MISMATCH,
                "grader 說多分支但證據不足以支撐多個分支，先改寫重試",
            )
        return DiagnosisOutcome(
            Diagnosis.OUT_OF_SCOPE,
            "重試後仍無足夠證據支撐任何分支",
        )

    # 6. grader 說措辭不匹配
    if grade.verdict == Diagnosis.LEXICAL_MISMATCH:
        if not retried_variants:
            return DiagnosisOutcome(Diagnosis.LEXICAL_MISMATCH, "grader 判定措辭不匹配，改寫後重試")
        if has_branch_evidence(signals):
            return DiagnosisOutcome(
                Diagnosis.MULTI_BRANCH,
                "改寫後仍不匹配，但多來源競爭，改以分支處理",
            )
        return DiagnosisOutcome(Diagnosis.OUT_OF_SCOPE, "改寫後仍找不到貼切內容")

    # 7. grader 說語料庫沒有
    if grade.verdict == Diagnosis.OUT_OF_SCOPE:
        if not retried_variants and signals.top1 >= config.SCORE_SUPPORTIVE:
            return DiagnosisOutcome(
                Diagnosis.LEXICAL_MISMATCH,
                f"grader 說無資料但 top1={signals.top1} 不算低，值得改寫一次再確認",
            )
        return DiagnosisOutcome(Diagnosis.OUT_OF_SCOPE, "grader 判定語料庫未涵蓋此主題")

    # 8. grader 失效：退回純數值規則，且偏保守
    return _diagnose_by_signals_only(signals, retried_variants)


def _diagnose_by_signals_only(signals: Signals, retried_variants: bool) -> DiagnosisOutcome:
    concentrated = signals.gap >= config.GAP_AMBIGUOUS or signals.max_source_share >= 0.6

    if signals.top1 >= config.SCORE_ANSWERABLE:
        if concentrated:
            return DiagnosisOutcome(
                Diagnosis.SUFFICIENT,
                "grader 失效；分數達門檻且證據集中",
                low_confidence=True,
            )
        if has_branch_evidence(signals):
            return DiagnosisOutcome(
                Diagnosis.MULTI_BRANCH,
                "grader 失效；分數達門檻但多來源競爭",
                low_confidence=True,
            )
        return DiagnosisOutcome(
            Diagnosis.SUFFICIENT,
            "grader 失效；分數達門檻",
            low_confidence=True,
        )

    if signals.top1 >= config.SCORE_SUPPORTIVE:
        if not retried_variants:
            return DiagnosisOutcome(
                Diagnosis.LEXICAL_MISMATCH,
                "grader 失效；分數中等，先改寫重試",
                low_confidence=True,
            )
        if has_branch_evidence(signals):
            return DiagnosisOutcome(
                Diagnosis.MULTI_BRANCH,
                "grader 失效；重試後多來源競爭",
                low_confidence=True,
            )

    return DiagnosisOutcome(
        Diagnosis.OUT_OF_SCOPE,
        "grader 失效；分數不足以支撐作答",
        low_confidence=True,
    )


def has_branch_evidence(signals: Signals) -> bool:
    """是否真的有多個分支各自具備證據。

    需要同時滿足：足夠多的 chunk 有證據強度、來源不只一個、且彼此分數接近
    （gap 小代表互相競爭；gap 大代表其實有明確勝者，不是分支問題）。
    """
    return (
        signals.n_supportive >= config.MULTI_BRANCH_MIN_STRONG
        and signals.n_sources >= 2
        and signals.gap < config.GAP_AMBIGUOUS
    )
