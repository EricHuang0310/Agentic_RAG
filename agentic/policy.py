"""分支處理策略：確定是多分支問題後，決定要「分情境全部列出」還是「反問」。

這是舊版最缺的一層。舊版只要分數低就反問，等於把反問當成 pipeline 的
入口閘門；正確的做法是把反問降級成最後手段，而且在候選少、答案短的
時候連反問都不必——直接分情境全部列出，省掉一輪往返，使用者還能順便
知道有這個區別存在。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

from . import config
from .schemas import Branch, Chunk, ClarificationOption, Decision


@dataclass
class BranchAction:
    decision: Decision
    reason: str
    clarification_question: str = ""
    options: List[ClarificationOption] = field(default_factory=list)
    dimension: str = ""

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

    # 同一個維度問過就不要再問，否則會出現「換句話問同一件事」的迴圈
    if dimension and dimension in asked_dimensions:
        return BranchAction(
            Decision.ANSWER_ALL_BRANCHES,
            f"維度「{dimension}」已經問過，改為分情境全部列出",
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
