"""診斷規則：核心是「LLM 提議、數值訊號否決」。"""

from agentic.diagnose import diagnose
from agentic.schemas import Diagnosis, GradeResult, Signals


def sig(**kw) -> Signals:
    base = dict(top1=2.0, top2=1.0, gap=1.0, n_chunks=6, n_supportive=3, n_answerable=2,
                n_sources=1, source_entropy=0.0, max_source_share=1.0)
    base.update(kw)
    return Signals(**base)


def grade(verdict: Diagnosis, **kw) -> GradeResult:
    return GradeResult(verdict=verdict, **kw)


def test_empty_retrieval_is_out_of_scope():
    out = diagnose(Signals(), grade(Diagnosis.SUFFICIENT), retried_variants=False, decomposed=False)
    assert out.diagnosis == Diagnosis.OUT_OF_SCOPE


def test_score_floor_vetoes_grader():
    """分數低於地板時，grader 說足夠也不採信。"""
    out = diagnose(
        sig(top1=-5.0), grade(Diagnosis.SUFFICIENT), retried_variants=True, decomposed=True
    )
    assert out.diagnosis == Diagnosis.OUT_OF_SCOPE


def test_sufficient_with_good_score():
    out = diagnose(sig(), grade(Diagnosis.SUFFICIENT), retried_variants=False, decomposed=False)
    assert out.diagnosis == Diagnosis.SUFFICIENT
    assert not out.low_confidence


def test_sufficient_but_low_score_retries_first():
    """grader 說足夠但分數未達門檻：先去找更強的證據，不要直接答。"""
    out = diagnose(
        sig(top1=0.5), grade(Diagnosis.SUFFICIENT), retried_variants=False, decomposed=False
    )
    assert out.diagnosis == Diagnosis.LEXICAL_MISMATCH


def test_sufficient_low_score_accepted_after_retry_as_low_confidence():
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
        sig(n_sources=1, gap=2.0), grade(Diagnosis.MULTI_BRANCH),
        retried_variants=False, decomposed=False,
    )
    assert out.diagnosis == Diagnosis.LEXICAL_MISMATCH


def test_multi_branch_confirmed_by_signals():
    out = diagnose(
        sig(top1=1.2, top2=1.1, gap=0.1, n_sources=3, n_supportive=3),
        grade(Diagnosis.MULTI_BRANCH),
        retried_variants=False,
        decomposed=False,
    )
    assert out.diagnosis == Diagnosis.MULTI_BRANCH


def test_lexical_mismatch_retries_once_then_converges():
    first = diagnose(
        sig(top1=0.5, gap=0.1, n_sources=3), grade(Diagnosis.LEXICAL_MISMATCH),
        retried_variants=False, decomposed=False,
    )
    assert first.diagnosis == Diagnosis.LEXICAL_MISMATCH

    second = diagnose(
        sig(top1=0.5, gap=0.1, n_sources=3), grade(Diagnosis.LEXICAL_MISMATCH),
        retried_variants=True, decomposed=False,
    )
    assert second.diagnosis == Diagnosis.MULTI_BRANCH  # 多來源競爭 -> 交給分支策略


def test_lexical_mismatch_after_retry_without_competition_refuses():
    out = diagnose(
        sig(top1=0.5, gap=2.0, n_sources=1), grade(Diagnosis.LEXICAL_MISMATCH),
        retried_variants=True, decomposed=False,
    )
    assert out.diagnosis == Diagnosis.OUT_OF_SCOPE


def test_out_of_scope_gets_one_rewrite_chance_when_score_not_low():
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
        sig(top1=3.0, gap=2.0, max_source_share=0.9), grade(Diagnosis.UNKNOWN),
        retried_variants=False, decomposed=False,
    )
    assert concentrated.diagnosis == Diagnosis.SUFFICIENT
    assert concentrated.low_confidence

    competing = diagnose(
        sig(top1=1.2, top2=1.15, gap=0.05, n_sources=3, n_supportive=3, max_source_share=0.4),
        grade(Diagnosis.UNKNOWN),
        retried_variants=False,
        decomposed=False,
    )
    assert competing.diagnosis == Diagnosis.MULTI_BRANCH

    weak = diagnose(
        sig(top1=-0.5, n_supportive=0), grade(Diagnosis.UNKNOWN),
        retried_variants=True, decomposed=False,
    )
    assert weak.diagnosis == Diagnosis.OUT_OF_SCOPE
