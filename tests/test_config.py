"""門檻設定的解析與預設值。"""

import importlib

import pytest

from agentic import config


def reload_config(monkeypatch, **env):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def restore_config():
    """測試會 reload config 模組，結束後還原成乾淨狀態。"""
    yield
    importlib.reload(config)


def test_defaults_match_the_measured_0_to_1_scale():
    assert config.SCORE_ANSWERABLE == 0.7   # 實測的反問切點
    assert config.SCORE_SUPPORTIVE == 0.35
    assert config.GAP_AMBIGUOUS == 0.08
    # 硬地板預設停用，拒答必須經過 grader
    assert config.SCORE_FLOOR == float("-inf")
    assert config.THRESHOLDS_CALIBRATED is True


def test_floor_can_be_enabled_by_env(monkeypatch):
    reloaded = reload_config(monkeypatch, ARKKB_SCORE_FLOOR="0.12")
    assert reloaded.SCORE_FLOOR == 0.12


def test_threshold_can_be_disabled_with_off(monkeypatch):
    reloaded = reload_config(monkeypatch, ARKKB_SCORE_FLOOR="off")
    assert reloaded.SCORE_FLOOR == float("-inf")


def test_calibration_can_be_turned_off(monkeypatch):
    reloaded = reload_config(monkeypatch, ARKKB_THRESHOLDS_CALIBRATED="0")
    assert reloaded.THRESHOLDS_CALIBRATED is False
