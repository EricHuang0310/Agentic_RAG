"""多變體檢索結果的正規化與融合。

知識庫 API 只吃單一 query 字串，沒有 BM25 或 hybrid 端點可用，因此
「hybrid」在這裡以 multi-query fan-out + RRF 的方式近似：同一個問題送出
數個不同表述（原問法、SOP 術語改寫、HyDE 假想條文、關鍵詞式查詢），
再用 Reciprocal Rank Fusion 融合排名。真正的 lexical 通道要等後端
提供 BM25 才能補上。
"""

import hashlib
from typing import Any, Dict, List, Sequence, Tuple

from . import config
from .schemas import Chunk


def parse_chunks(raw_items: Sequence[Dict[str, Any]]) -> List[Chunk]:
    """把知識庫 API 的原始 item 轉成 Chunk。分數欄位位置沿用舊版的多層 fallback。"""
    chunks: List[Chunk] = []
    for item in raw_items:
        score = item.get("kwargs", {}).get("relevance_score")
        if score is None:
            score = item.get("relevance_score", item.get("score", 0.0))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        content = (item.get("content") or "").strip()
        filename = item.get("filename") or "未知檔案"
        page = str(item.get("page") or "")
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(filename, page, content),
                filename=filename,
                page=page,
                score=score,
                content=content,
            )
        )
    return chunks


def make_chunk_id(filename: str, page: str, content: str) -> str:
    """以內容雜湊為主的識別碼，讓不同 query 變體命中的同一段落可以對齊去重。"""
    digest = hashlib.sha1(f"{filename}|{page}|{content}".encode("utf-8")).hexdigest()
    return f"c_{digest[:12]}"


def rrf_fuse(
    variant_results: Sequence[Tuple[str, List[Chunk]]],
    k: int = config.RRF_K,
    top_n: int = config.TOP_N,
) -> List[Chunk]:
    """Reciprocal Rank Fusion。

    variant_results: [(變體名稱, 該變體的 chunk 清單（已依分數排序）), ...]
    去重後每個 chunk 的 score 取各變體的最大值（代表最佳證據強度），
    排序則依 RRF 分數（代表跨變體的一致性）。
    """
    fused: Dict[str, Chunk] = {}
    rrf: Dict[str, float] = {}

    for variant_name, chunks in variant_results:
        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        for rank, chunk in enumerate(ranked, start=1):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            existing = fused.get(chunk.chunk_id)
            if existing is None:
                fused[chunk.chunk_id] = chunk.model_copy(
                    update={"matched_variants": [variant_name]}
                )
            else:
                if chunk.score > existing.score:
                    existing.score = chunk.score
                if variant_name not in existing.matched_variants:
                    existing.matched_variants.append(variant_name)

    for chunk_id, chunk in fused.items():
        chunk.rrf_score = round(rrf.get(chunk_id, 0.0), 6)

    ordered = sorted(fused.values(), key=lambda c: (c.rrf_score, c.score), reverse=True)
    return ordered[:top_n]
