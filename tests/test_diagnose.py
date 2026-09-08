"""診斷規則：核心是「LLM 提議、數值訊號否決」。

絕對門檻只有在 THRESHOLDS_CALIBRATED=True 時才會用來否決 grader，
所以驗證訊號否決的測試都要掛上 calibrated fixture；
未校準（預設）模式另有一組測試，確認不會單憑絕對分數拒答。
"""

import pytest

from agentic import config
from agentic.diagnose import diagnose
from agentic.schemas import Diagnosis, GradeResult, Signals


@pytest.fixture
def calibrated(monkeypatch):
    """模擬「門檻已針對這個 reranker 校準過」的正式環境（預設就是這個模式）。"""
    monkeypatch.setattr(config, "THRESHOLDS_CALIBRATED", True)


@pytest.fixture
def uncalibrated(monkeypatch):
    """模擬尚未校準：不允許單憑絕對分數拒答。"""
    monkeypatch.setattr(config, "THRESHOLDS_CALIBRATED", False)


@pytest.fixture
def floor_enabled(monkeypatch):
    """SCORE_FLOOR 預設停用，要驗證硬地板否決得先給它一個實際值。"""
    monkeypatch.setattr(config, "THRESHOLDS_CALIBRATED", True)
    monkeypatch.setattr(config, "SCORE_FLOOR", 0.12)


def sig(**kw) -> Signals:
    """預設是「單一來源、分數明確」的證據充足情況（0～1 尺度）。"""
    base = dict(top1=0.85, top2=0.45, gap=0.40, n_chunks=6, n_supportive=3, n_answerable=2,
                n_sources=1, source_entropy=0.0, max_source_share=1.0)
    base.update(kw)
    return Signals(**base)


def grade(verdict: Diagnosis, **kw) -> GradeResult:
    return GradeResult(verdict=verdict, **kw)


def test_empty_retrieval_is_out_of_scope():
    out = diagnose(Signals(), grade(Diagnosis.SUFFICIENT), retried_variants=False, decomposed=False)
    assert out.diagnosis == Diagnosis.OUT_OF_SCOPE


def test_score_floor_vetoes_grader(floor_enabled):
    """分數低於地板時，grader 說足夠也不採信。"""
    out = diagnose(
        sig(top1=0.05), grade(Diagnosis.SUFFICIENT), retried_variants=True, decomposed=True
    )
    assert out.diagnosis == Diagnosis.OUT_OF_SCOPE
    assert "地板" in out.reason


def test_disabled_floor_never_vetoes(calibrated):
    """預設停用地板：極低分時的拒答必須由後續規則決定，不是地板直接判死。"""
    out = diagnose(
        sig(top1=0.05), grade(Diagnosis.SUFFICIENT), retried_variants=False, decomposed=True
    )
    assert out.diagnosis == Diagnosis.LEXICAL_MISMATCH  # 先去改寫，不是直接拒答


def test_sufficient_with_good_score(calibrated):
    out = diagnose(sig(), grade(Diagnosis.SUFFICIENT), retried_variants=False, decomposed=False)
    assert out.diagnosis == Diagnosis.SUFFICIENT
    assert not out.low_confidence


def test_sufficient_but_low_score_retries_first(calibrated):
    """grader 說足夠但分數未達門檻：先去找更強的證據，不要直接答。"""
    out = diagnose(
        sig(top1=0.5), grade(Diagnosis.SUFFICIENT), retried_variants=False, decomposed=False
    )
    assert out.diagnosis == Diagnosis.LEXICAL_MISMATCH


def test_sufficient_low_score_accepted_after_retry_as_low_confidence(calibrated):
    out = diagnose(
        sig(top1=0.5), grade(Diagnosis.SUFFICIENT), retried_variants=True, decomposed=False
    )
    assert out.diagnosis == Diagnosis.SUFFICIENT
    assert out.low_confidence


def test_compound_takes_precedence_over_retry():
    out = diagnose(
        sig(top1=0.5),
        grade(Diagnosis.LEXICAL_MISMATCH),
        retried_variants=False,
        decomposed=False,
        is_compound=True,
    )
    assert out.diagnosis == Diagnosis.COMPOUND


def test_compound_not_repeated_after_decomposition():
    out = diagnose(
        sig(top1=0.5),
        grade(Diagnosis.LEXICAL_MISMATCH),
        retried_variants=False,
        decomposed=True,
        is_compound=True,
    )
    assert out.diagnosis == Diagnosis.LEXICAL_MISMATCH


def test_multi_branch_needs_real_evidence():
    """grader 說多分支，但只有單一來源、gap 大 -> 不是分支問題。"""
    out = diagnose(
        sig(n_sources=1, gap=0.4), grade(Diagnosis.MULTI_BRANCH),
        retried_variants=False, decomposed=False,
    )
    assert out.diagnosis == Diagnosis.LEXICAL_MISMATCH


def test_multi_branch_confirmed_by_signals():
    out = diagnose(
        sig(top1=0.72, top2=0.68, gap=0.04, n_sources=3, n_supportive=3),
        grade(Diagnosis.MULTI_BRANCH),
        retried_variants=False,
        decomposed=False,
    )
    assert out.diagnosis == Diagnosis.MULTI_BRANCH


def test_lexical_mismatch_retries_once_then_converges():
    first = diagnose(
        sig(top1=0.5, gap=0.04, n_sources=3), grade(Diagnosis.LEXICAL_MISMATCH),
        retried_variants=False, decomposed=False,
    )
    assert first.diagnosis == Diagnosis.LEXICAL_MISMATCH

    second = diagnose(
        sig(top1=0.5, gap=0.04, n_sources=3), grade(Diagnosis.LEXICAL_MISMATCH),
        retried_variants=True, decomposed=False,
    )
    assert second.diagnosis == Diagnosis.MULTI_BRANCH  # 多來源競爭 -> 交給分支策略


def test_lexical_mismatch_after_retry_without_competition_refuses():
    out = diagnose(
        sig(top1=0.5, gap=0.4, n_sources=1), grade(Diagnosis.LEXICAL_MISMATCH),
        retried_variants=True, decomposed=False,
    )
    assert out.diagnosis == Diagnosis.OUT_OF_SCOPE


def test_out_of_scope_gets_one_rewrite_chance_when_score_not_low(calibrated):
    out = diagnose(
        sig(top1=0.8), grade(Diagnosis.OUT_OF_SCOPE), retried_variants=False, decomposed=False
    )
    assert out.diagnosis == Diagnosis.LEXICAL_MISMATCH


def test_out_of_scope_confirmed_after_retry():
    out = diagnose(
        sig(top1=0.8), grade(Diagnosis.OUT_OF_SCOPE), retried_variants=True, decomposed=False
    )
    assert out.diagnosis == Diagnosis.OUT_OF_SCOPE


def test_grader_failure_falls_back_to_signals_and_flags_low_confidence():
    concentrated = diagnose(
        sig(top1=0.9, gap=0.4, max_source_share=0.9), grade(Diagnosis.UNKNOWN),
        retried_variants=False, decomposed=False,
    )
    assert concentrated.diagnosis == Diagnosis.SUFFICIENT
    assert concentrated.low_confidence

    competing = diagnose(
        sig(top1=0.9, top2=0.87, gap=0.03, n_sources=3, n_supportive=3, max_source_share=0.4),
        grade(Diagnosis.UNKNOWN),
        retried_variants=False,
        decomposed=False,
    )
    assert competing.diagnosis == Diagnosis.MULTI_BRANCH

    weak = diagnose(
        sig(top1=0.05, n_supportive=0), grade(Diagnosis.UNKNOWN),
        retried_variants=True, decomposed=False,
    )
    assert weak.diagnosis == Diagnosis.OUT_OF_SCOPE


# ==================== 未校準模式（預設）====================
def test_uncalibrated_mode_does_not_refuse_on_absolute_score_alone(uncalibrated):
    """門檻沒校準過時，不可單憑「分數很低」就拒答——那個門檻只是猜測。"""
    out = diagnose(
        sig(top1=0.05), grade(Diagnosis.SUFFICIENT), retried_variants=False, decomposed=False
    )
    assert out.diagnosis == Diagnosis.SUFFICIENT
    assert out.low_confidence  # 採信 grader，但標記低信心


def test_uncalibrated_mode_still_refuses_when_grader_says_not_in_corpus(uncalibrated):
    """把關改由 grader 負責：它說語料庫沒有，重試過後仍然拒答。"""
    out = diagnose(
        sig(top1=0.05), grade(Diagnosis.OUT_OF_SCOPE), retried_variants=True, decomposed=False
    )
    assert out.diagnosis == Diagnosis.OUT_OF_SCOPE


def test_uncalibrated_mode_retries_before_refusing(uncalibrated):
    out = diagnose(
        sig(top1=0.05), grade(Diagnosis.OUT_OF_SCOPE), retried_variants=False, decomposed=False
    )
    assert out.diagnosis == Diagnosis.LEXICAL_MISMATCH


def test_empty_retrieval_still_refuses_when_uncalibrated():
    """檢索完全沒東西與分數尺度無關，仍然該拒答。"""
    out = diagnose(
        Signals(), grade(Diagnosis.SUFFICIENT), retried_variants=False, decomposed=False
    )
    assert out.diagnosis == Diagnosis.OUT_OF_SCOPE
