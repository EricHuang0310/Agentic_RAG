"""探測 reranker 的分數尺度，作為設定閾值的第一步。

agentic/config.py 裡的 SCORE_ANSWERABLE / SCORE_SUPPORTIVE / SCORE_FLOOR
是絕對門檻，只對「同一個 reranker + 同一批語料」有意義。在還沒看過真實分數
分布以前，那些預設值都只是猜測，猜錯的代價是把答得出來的問題判成拒答。

用法：

    # 直接給問題
    python scripts/probe_scores.py "遺留物現金怎麼處理？" "銷戶要怎麼辦理？"

    # 或給一個每行一題的檔案；標註過的話用 tab 分隔第二欄
    #   遺留物現金怎麼處理？<TAB>answer
    #   信用卡分期利率？<TAB>refuse
    python scripts/probe_scores.py --file questions.txt

有標註時會分別列出各類問題的分數分布，並建議切點。
"""

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic import config  # noqa: E402
from agentic.fusion import parse_chunks  # noqa: E402


def summarize(scores: Sequence[float]) -> Dict[str, float]:
    """一組 top1 分數的摘要統計。"""
    if not scores:
        return {}
    ordered = sorted(scores)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "p25": round(_percentile(ordered, 0.25), 4),
        "median": round(_percentile(ordered, 0.50), 4),
        "p75": round(_percentile(ordered, 0.75), 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
    }


def _percentile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def suggest_thresholds(
    by_label: Dict[str, List[float]]
) -> Dict[str, Optional[float]]:
    """依標註資料建議切點。

    - answerable：能直接回答的問題，其 top1 大致落在哪個下緣（取 p25）
    - floor：該拒答的問題，其 top1 大致落在哪個上緣（取 p75）
    - supportive：兩者之間

    只是起點，實際仍要看兩類分布重疊多少再手動微調。
    """
    answerable = sorted(by_label.get("answer", []))
    refusable = sorted(by_label.get("refuse", []))

    suggestion: Dict[str, Optional[float]] = {
        "SCORE_ANSWERABLE": None,
        "SCORE_SUPPORTIVE": None,
        "SCORE_FLOOR": None,
    }
    if answerable:
        suggestion["SCORE_ANSWERABLE"] = round(_percentile(answerable, 0.25), 4)
    if refusable:
        suggestion["SCORE_FLOOR"] = round(_percentile(refusable, 0.75), 4)
    if answerable and refusable:
        suggestion["SCORE_SUPPORTIVE"] = round(
            (suggestion["SCORE_ANSWERABLE"] + suggestion["SCORE_FLOOR"]) / 2, 4
        )
    return suggestion


def load_questions(path: Path) -> List[Tuple[str, str]]:
    """讀取問題檔，回傳 [(問題, 標註)]。沒有標註時標註為空字串。"""
    rows: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        rows.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
    return rows


def probe(questions: Sequence[Tuple[str, str]]) -> None:
    import httpx

    by_label: Dict[str, List[float]] = {}
    all_scores: List[float] = []

    with httpx.Client(timeout=config.RETRIEVE_TIMEOUT) as client:
        for question, label in questions:
            payload = {
                "query": question,
                "kb_list": config.TARGET_KB_LIST,
                "reranker_model": config.RERANKER_MODEL,
                "top_n": config.TOP_N,
                "kwargs": {"parent_vector": {"top_k": config.TOP_K}},
            }
            try:
                response = client.post(
                    config.RETRIEVE_URL,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                chunks = parse_chunks(response.json().get("data") or [])
            except Exception as exc:  # noqa: BLE001
                print(f"✗ {question[:40]}  檢索失敗：{exc}")
                continue

            if not chunks:
                print(f"· {question[:40]}  檢索無結果")
                continue

            scores = sorted((c.score for c in chunks), reverse=True)
            sources = {c.filename for c in chunks}
            all_scores.append(scores[0])
            if label:
                by_label.setdefault(label, []).append(scores[0])
            tag = f"[{label}] " if label else ""
            print(
                f"· {tag}{question[:40]}\n"
                f"    top1={scores[0]:.4f}  top2={scores[1]:.4f}  "
                f"gap={scores[0] - scores[1]:.4f}  來源數={len(sources)}"
                if len(scores) > 1
                else f"· {tag}{question[:40]}\n    top1={scores[0]:.4f}  （只有 1 個 chunk）"
            )

    print("\n" + "=" * 60)
    print("整體 top1 分布：", summarize(all_scores))
    for label, scores in sorted(by_label.items()):
        print(f"  標註「{label}」：", summarize(scores))

    if by_label:
        print("\n建議起點（仍需依兩類分布的重疊程度手動微調）：")
        for name, value in suggest_thresholds(by_label).items():
            print(f"  {name} = {value}")
        print("\n調好之後設定 ARKKB_THRESHOLDS_CALIBRATED=1 才會啟用訊號否決。")
    else:
        print("\n提示：在問題後面用 tab 加上 answer / clarify / refuse 標註，可以得到建議切點。")


def main() -> None:
    parser = argparse.ArgumentParser(description="探測 reranker 分數尺度")
    parser.add_argument("questions", nargs="*", help="要探測的問題")
    parser.add_argument("--file", type=Path, help="每行一題的檔案，可用 tab 加標註")
    args = parser.parse_args()

    rows: List[Tuple[str, str]] = [(q, "") for q in args.questions]
    if args.file:
        rows.extend(load_questions(args.file))
    if not rows:
        parser.error("請至少提供一個問題，或用 --file 指定問題檔")

    print(f"知識庫：{config.TARGET_KB_LIST}")
    print(f"Reranker：{config.RERANKER_MODEL}")
    print(f"目前門檻：answerable={config.SCORE_ANSWERABLE} "
          f"supportive={config.SCORE_SUPPORTIVE} floor={config.SCORE_FLOOR} "
          f"calibrated={config.THRESHOLDS_CALIBRATED}\n")
    probe(rows)


if __name__ == "__main__":
    main()
