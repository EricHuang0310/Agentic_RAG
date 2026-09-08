"""從檢索結果算出數值訊號。

舊版只用 max(scores) 一個標量做決策，無法區分「問題模糊」與「語料庫沒有」。
這裡多算幾個同一次檢索就拿得到的免費特徵，其中最關鍵的是
來源分散度（source_entropy / max_source_share）：

  - 高分且集中在單一檔案       -> 證據明確
  - 高分但散在多個檔案且 gap 小 -> 多分支互斥，問題模糊
  - 全體低分且散在多個檔案     -> 語料庫沒有這個主題
"""

import math
from typing import Dict, List

from . import config
from .schemas import Chunk, Signals


def compute_signals(chunks: List[Chunk]) -> Signals:
    if not chunks:
        return Signals()

    scores = sorted((c.score for c in chunks), reverse=True)
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    top1 = scores[0]
    top2 = scores[1] if n > 1 else scores[0]

    weights = _softmax(scores)
    source_weight: Dict[str, float] = {}
    for chunk, weight in zip(sorted(chunks, key=lambda c: c.score, reverse=True), weights):
        source_weight[chunk.filename] = source_weight.get(chunk.filename, 0.0) + weight

    return Signals(
        top1=round(top1, 4),
        top2=round(top2, 4),
        gap=round(top1 - top2, 4),
        mean=round(mean, 4),
        std=round(math.sqrt(variance), 4),
        n_chunks=n,
        n_supportive=sum(1 for s in scores if s >= config.SCORE_SUPPORTIVE),
        n_answerable=sum(1 for s in scores if s >= config.SCORE_ANSWERABLE),
        n_sources=len(source_weight),
        source_entropy=round(_normalized_entropy(list(source_weight.values())), 4),
        max_source_share=round(max(source_weight.values()), 4),
    )


def _softmax(scores: List[float]) -> List[float]:
    """把 reranker 分數轉成權重。

    用 softmax 而非「除以總和」，因為 cross-encoder 分數可能為負，
    直接歸一化會產生負權重與除零問題。
    """
    if not scores:
        return []
    peak = max(scores)
    exps = [math.exp(s - peak) for s in scores]
    total = sum(exps)
    return [e / total for e in exps] if total else [1.0 / len(scores)] * len(scores)


def _normalized_entropy(weights: List[float]) -> float:
    """正規化 Shannon 熵：0 = 完全集中在單一來源，1 = 均勻分散。"""
    if len(weights) <= 1:
        return 0.0
    total = sum(weights)
    if total <= 0:
        return 0.0
    probs = [w / total for w in weights if w > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(weights))
