"""分支處理策略：確定是多分支問題後，決定要「分情境全部列出」還是「反問」。

這是舊版最缺的一層。舊版只要分數低就反問，等於把反問當成 pipeline 的
入口閘門；正確的做法是把反問降級成最後手段，而且在候選少、答案短的
時候連反問都不必——直接分情境全部列出，省掉一輪往返，使用者還能順便
知道有這個區別存在。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from . import config
from .schemas import Branch, Chunk, ClarificationOption, Decision


@dataclass
class BranchAction:
    decision: Decision
    reason: str
    clarification_question: str = ""
    options: List[ClarificationOption] = field(default_factory=list)
    dimension: str = ""
    branch_labels: List[str] = field(default_factory=list)

    def as_detail(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "dimension": self.dimension,
            "options": [o.label for o in self.options],
        }


def choose_branch_action(
    branches: Sequence[Branch],
    chunks: Sequence[Chunk],
    dimension: str = "",
    *,
    asked_dimensions: Set[str] | None = None,
    asked_branch_labels: Sequence[str] | None = None,
    clarification_count: int = 0,
) -> BranchAction:
    """決定多分支情況下的行動。"""
    asked_dimensions = asked_dimensions or set()

    if not branches:
        return BranchAction(Decision.REFUSE, "判定為多分支但抽不出任何分支")

    # 反問次數已達上限：不再問，改成分情境全答（有內容總比空手好）
    if clarification_count >= config.MAX_CLARIFICATIONS:
        return BranchAction(
            Decision.ANSWER_ALL_BRANCHES,
            f"已反問 {clarification_count} 次達上限，改為分情境全部列出",
            dimension=dimension,
        )

    # 同一個維度問過就不要再問，否則會出現「換句話問同一件事」的迴圈。
    # 比對前先正規化，否則「業務類別」與「業務類別 」之類的差異會讓防護失效。
    if dimension and normalize_label(dimension) in {
        normalize_label(d) for d in asked_dimensions
    }:
        return BranchAction(
            Decision.ANSWER_ALL_BRANCHES,
            f"維度「{dimension}」已經問過，改為分情境全部列出",
            dimension=dimension,
        )

    # 維度名稱由 LLM 生成，兩輪之間可能改名（「業務類別」→「申請類型」），
    # 只靠維度比對會漏掉重複反問。分支標籤來自同一批文件相對穩定，用它再擋一層。
    if _labels_already_asked(branches, asked_branch_labels):
        return BranchAction(
            Decision.ANSWER_ALL_BRANCHES,
            "相同的分支已經問過（維度名稱不同但選項重複），改為分情境全部列出",
            dimension=dimension,
        )

    supporting_chars = _supporting_chars(branches, chunks)
    if len(branches) <= config.INLINE_BRANCH_MAX and supporting_chars <= config.INLINE_BRANCH_MAX_CHARS:
        return BranchAction(
            Decision.ANSWER_ALL_BRANCHES,
            f"分支數 {len(branches)} 且支撐內容 {supporting_chars} 字，直接分情境全部列出",
            dimension=dimension,
        )

    return BranchAction(
        Decision.CLARIFY,
        f"分支數 {len(branches)}、支撐內容 {supporting_chars} 字，全列會過長，改為反問",
        clarification_question=build_clarification_question(branches, dimension),
        options=build_options(branches),
        dimension=dimension,
        branch_labels=[b.label for b in branches],
    )


def build_clarification_question(branches: Sequence[Branch], dimension: str = "") -> str:
    """組出反問話術。

    內容完全由檢索結果決定，不再是寫死的字串——換知識庫或新增業務類別時
    不需要改任何程式碼。
    """
    labels = "、".join(f"「{b.label}」" for b in branches)
    if dimension:
        return (
            f"您的問題在作業規範中依{dimension}有不同的處理方式，"
            f"包含 {labels}。請問您要辦理的屬於哪一種？"
        )
    return f"您的問題在作業規範中有幾種不同的處理方式：{labels}。請問您指的是哪一種？"


def build_options(branches: Sequence[Branch]) -> List[ClarificationOption]:
    """把分支轉成結構化選項，前端可直接渲染成按鈕。

    使用者點選比自由輸入準得多，也讓第二輪的 query 重構不必猜使用者的意思。
    """
    options = [
        ClarificationOption(label=b.label, value=_option_value(b))
        for b in branches
    ]
    options.append(ClarificationOption(label="以上皆非／我要描述我的情況", value="__other__"))
    return options


# 括號內容（選項 value 帶的關鍵詞）與標點空白，比對標籤時一律忽略
_PARENTHETICAL = re.compile(r"[（(][^)）]*[)）]")
_NOISE = re.compile(r"[\s、,，。.：:；;「」『』\"\'!！?？~～\-_/]+")


def normalize_label(text: str) -> str:
    """把標籤正規化成可比對的形式（去掉括號內容、標點與空白）。"""
    return _NOISE.sub("", _PARENTHETICAL.sub("", text or "")).lower()


def resolve_branch_from_message(
    message: str, branches: Sequence[Branch]
) -> Optional[Branch]:
    """判斷使用者這句話是不是已經指定了某一個分支。

    這是**不依賴 session** 的防護。session 若因為前端沒帶回 session_id、
    TTL 過期或服務重啟而遺失，靠 asked_dimensions 的重複反問防護會整個失效，
    使用者就會看到一模一樣的問題再問一次。這裡改從訊息內容本身判斷：
    使用者的話若明確對應到其中一個分支，就直接回答那個分支，不再問。

    必須「恰好一個」分支匹配才算數；像「變更負責人」同時命中
    「獨資戶變更負責人」與「公司變更負責人」時仍屬模糊，該問還是要問。
    """
    normalized = normalize_label(message)
    if len(normalized) < 2:
        return None

    matches = []
    for branch in branches:
        label = normalize_label(branch.label)
        if not label:
            continue
        if label == normalized or label in normalized or normalized in label:
            matches.append(branch)
    return matches[0] if len(matches) == 1 else None


def _labels_already_asked(
    branches: Sequence[Branch], asked_branch_labels: Sequence[str] | None
) -> bool:
    """這次要問的分支，是不是和上次問過的大致相同。"""
    if not asked_branch_labels:
        return False
    current = {normalize_label(b.label) for b in branches if normalize_label(b.label)}
    previous = {normalize_label(l) for l in asked_branch_labels if normalize_label(l)}
    if not current or not previous:
        return False
    return len(current & previous) / len(current) >= 0.6


def _option_value(branch: Branch) -> str:
    """選項回傳值帶上關鍵詞，讓下一輪檢索拿到更多可用訊號。"""
    if branch.key_terms:
        return f"{branch.label}（{'、'.join(branch.key_terms)}）"
    return branch.label


def _supporting_chars(branches: Sequence[Branch], chunks: Sequence[Chunk]) -> int:
    """估算「分情境全部列出」的答案長度：所有分支引用到的 chunk 內容總字數。"""
    by_id: Dict[str, Chunk] = {c.chunk_id: c for c in chunks}
    counted: Set[str] = set()
    total = 0
    for branch in branches:
        for chunk_id in branch.chunk_ids:
            if chunk_id in counted:
                continue
            counted.add(chunk_id)
            chunk = by_id.get(chunk_id)
            if chunk:
                total += len(chunk.content)
    # 分支沒有標註 chunk 時，保守以全部 chunk 估算
    if total == 0:
        total = sum(len(c.content) for c in chunks)
    return total
