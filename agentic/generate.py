"""回答生成。

保留舊版那套極嚴格的 system prompt（一字不漏、禁用概括詞、誠實不編造），
只多加一條強制標註出處的規則，並在生成後用程式驗證引用編號是否存在。

這裡只做「引用編號有效性」這種零成本的檢查，不做句級 grounding / NLI 驗證；
後者需要額外一次以上的模型呼叫，屬於下一階段。
"""

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .budget import Budget
from .clients import LLMClient
from .schemas import Branch, Chunk

_CITATION = re.compile(r"\[文獻\s*(\d+)\]")

_BASE_RULES = (
    "你是一個極度嚴格且專業的內部作業流程助手。請「完全且僅能」根據下方提供的【參考資訊】"
    "與【使用者問題】來回答使用者目前的提問。\n\n"
    "在回答時，請務必嚴格遵守以下規則：\n"
    "1. 一字不漏：當使用者詢問任何方案、步驟、檢核項目或規定時，必須將參考資訊中的相關項目"
    "完整列出，切勿省略關鍵條件。\n"
    "2. 禁用概括詞：絕對禁止在回答中使用「等」、「等等」、「包含但不限於」、「以及其他...」"
    "等任何帶有省略或模糊意味的詞彙。\n"
    "3. 條列清晰：依照作業流程的順序或分類使用條列式呈現。\n"
    "4. 誠實不編造：如果參考資訊中沒有答案或缺少條件，請直接告知「參考資訊中未提供此作業流程說明」，"
    "絕對不可以自行猜測。\n"
    "5. 語言要求：請全程使用繁體中文回答。\n"
    "6. 標註出處：每一個條列項目的結尾都必須標註它依據哪一段參考資訊，格式為 [文獻 N]，"
    "N 是參考資訊的編號。不可以標註不存在的編號。\n"
)

_SINGLE_SYSTEM = _BASE_RULES

_MULTI_BRANCH_SYSTEM = (
    _BASE_RULES
    + "\n"
    + "額外的重要規則：使用者的問題沒有指明情境，而參考資訊涵蓋多種互斥的情境。\n"
    "7. 分情境作答：請為每一種情境各自獨立作答，用「若屬〈情境名稱〉：」作為每一段的開頭，"
    "再條列該情境的流程。不要把不同情境的步驟混在一起。\n"
    "8. 開場提醒：回答的第一句話請簡短說明這個問題依情境不同而有不同處理方式，"
    "請使用者確認自己屬於哪一種。\n"
    "9. 不要偏袒：不可以自行猜測使用者屬於哪一種情境，也不可以只回答其中一種。\n"
)


@dataclass
class GeneratedAnswer:
    text: str
    invalid_citations: List[int]
    cited_indices: List[int]

    def as_detail(self) -> dict:
        return {
            "cited": self.cited_indices,
            "invalid_citations": self.invalid_citations,
            "length": len(self.text),
        }


def build_context(chunks: Sequence[Chunk]) -> str:
    """組出【參考資訊】。編號與 reference_chunks 的順序一致，方便前端對照引用。"""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        if not chunk.content:
            continue
        page = f"，頁次：{chunk.page}" if chunk.page else ""
        parts.append(f"[文獻 {index}]（來源：{chunk.filename}{page}）\n{chunk.content}")
    return "\n\n".join(parts)


async def generate_answer(
    query: str,
    chunks: Sequence[Chunk],
    llm: LLMClient,
    budget: Budget,
) -> GeneratedAnswer | None:
    """單一情境的回答。"""
    context = build_context(chunks)
    user_prompt = f"【使用者問題】：\n{query}\n\n【參考資訊】：\n{context}"
    raw = await llm.chat(_SINGLE_SYSTEM, user_prompt, budget)
    if raw is None:
        return None
    return _validate(raw, chunks)


async def generate_branch_answer(
    query: str,
    chunks: Sequence[Chunk],
    branches: Sequence[Branch],
    dimension: str,
    llm: LLMClient,
    budget: Budget,
) -> GeneratedAnswer | None:
    """分情境全部列出的回答，取代一次不必要的反問往返。"""
    context = build_context(chunks)
    branch_hint = "\n".join(
        f"- {b.label}" + (f"（關鍵詞：{'、'.join(b.key_terms)}）" if b.key_terms else "")
        for b in branches
    )
    dimension_hint = f"（區辨維度：{dimension}）" if dimension else ""
    user_prompt = (
        f"【使用者問題】：\n{query}\n\n"
        f"【需要分別作答的情境】{dimension_hint}：\n{branch_hint}\n\n"
        f"【參考資訊】：\n{context}"
    )
    raw = await llm.chat(_MULTI_BRANCH_SYSTEM, user_prompt, budget)
    if raw is None:
        return None
    return _validate(raw, chunks)


def _validate(raw: str, chunks: Sequence[Chunk]) -> GeneratedAnswer:
    text = raw.strip()
    cited, invalid = check_citations(text, len(chunks))
    if invalid:
        # 不擅自改寫回答內容，只在結尾附上提醒並記進 trace，讓人看得見問題
        text += (
            "\n\n（系統提醒：本回答中的出處標註 "
            + "、".join(f"[文獻 {i}]" for i in invalid)
            + " 不存在於參考資訊中，請以參考資訊原文為準。）"
        )
    return GeneratedAnswer(text=text, invalid_citations=invalid, cited_indices=cited)


def check_citations(text: str, n_contexts: int) -> Tuple[List[int], List[int]]:
    """回傳 (有效引用編號, 無效引用編號)。純字串檢查，零額外成本。"""
    found = [int(m) for m in _CITATION.findall(text)]
    valid = sorted({i for i in found if 1 <= i <= n_contexts})
    invalid = sorted({i for i in found if i < 1 or i > n_contexts})
    return valid, invalid


def build_references_summary(chunks: Sequence[Chunk]) -> str:
    """去重後的來源清單，格式與舊版一致。"""
    seen = set()
    lines = []
    for chunk in chunks:
        label = (
            f"檔案名稱: {chunk.filename} (頁數: {chunk.page})"
            if chunk.page
            else f"檔案名稱: {chunk.filename}"
        )
        if label not in seen:
            seen.add(label)
            lines.append(label)
    return "\n".join(lines)
