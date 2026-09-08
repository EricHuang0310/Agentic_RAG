"""校準輔助腳本的純計算部分。"""

from probe_scores import load_questions, suggest_thresholds, summarize


def test_summarize_basic_stats():
    stats = summarize([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["median"] == 2.5


def test_summarize_empty():
    assert summarize([]) == {}


def test_summarize_single_value():
    stats = summarize([2.5])
    assert stats["min"] == stats["max"] == stats["median"] == 2.5


def test_suggest_thresholds_orders_cut_points():
    suggestion = suggest_thresholds({
        "answer": [2.0, 2.5, 3.0, 3.5],
        "refuse": [-4.0, -3.0, -2.5, -2.0],
    })
    assert suggestion["SCORE_FLOOR"] < suggestion["SCORE_SUPPORTIVE"] < suggestion["SCORE_ANSWERABLE"]


def test_suggest_thresholds_handles_missing_labels():
    suggestion = suggest_thresholds({"answer": [1.0, 2.0]})
    assert suggestion["SCORE_ANSWERABLE"] is not None
    assert suggestion["SCORE_FLOOR"] is None
    assert suggestion["SCORE_SUPPORTIVE"] is None


def test_load_questions_parses_labels_and_skips_comments(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text(
        "# 註解\n\n遺留物現金怎麼處理？\t answer \n信用卡分期利率？\trefuse\n沒有標註的問題\n",
        encoding="utf-8",
    )
    rows = load_questions(path)
    assert rows == [
        ("遺留物現金怎麼處理？", "answer"),
        ("信用卡分期利率？", "refuse"),
        ("沒有標註的問題", ""),
    ]
