"""거래량-가격 괴리(매집 탐지 보조 신호) 계약 테스트."""

from market_briefing.scan_engine import (
    VPD_BASELINE_DAYS,
    VPD_MAX_ABS_PRICE_CHANGE_PCT,
    VPD_MIN_VOLUME_RATIO,
    VPD_RECENT_DAYS,
    build_snapshot_from_ohlcv,
    detect_volume_price_divergence,
)


def _series(recent_volume=300.0, latest_price=102.0):
    # 60일 기준 구간 + 20일 최근 구간. 가격 수익률의 기준은 최근 구간 직전 종가다.
    closes = [100.0] * 80
    closes[-1] = latest_price
    volumes = [100.0] * 60 + [recent_volume] * 20
    return closes, volumes


def test_detects_threefold_volume_with_price_inside_three_percent_band():
    closes, volumes = _series(recent_volume=300.0, latest_price=103.0)

    result = detect_volume_price_divergence(closes, volumes)

    assert result["available"] is True
    assert result["is_match"] is True
    assert result["volume_ratio"] == 3.0
    assert result["price_change_pct"] == 3.0
    assert result["volume_condition"] is True
    assert result["price_condition"] is True


def test_rejects_volume_spike_when_price_moves_outside_band():
    closes, volumes = _series(recent_volume=400.0, latest_price=103.01)

    result = detect_volume_price_divergence(closes, volumes)

    assert result["volume_condition"] is True
    assert result["price_condition"] is False
    assert result["is_match"] is False
    assert result["reason"] == "가격 변동폭 기준 미충족"


def test_rejects_flat_price_without_threefold_volume():
    closes, volumes = _series(recent_volume=299.0, latest_price=100.0)

    result = detect_volume_price_divergence(closes, volumes)

    assert result["volume_condition"] is False
    assert result["price_condition"] is True
    assert result["is_match"] is False


def test_baseline_excludes_the_recent_month():
    closes, _ = _series()
    # 최근 20일의 큰 거래량이 기준 평균에 섞였다면 비율은 3배보다 작아진다.
    volumes = [100.0] * 60 + [600.0] * 20

    result = detect_volume_price_divergence(closes, volumes)

    assert result["baseline_avg_volume"] == 100.0
    assert result["recent_avg_volume"] == 600.0
    assert result["volume_ratio"] == 6.0
    assert result["is_match"] is True


def test_reports_data_gap_instead_of_guessing_with_short_history():
    result = detect_volume_price_divergence([100.0] * 79, [100.0] * 79)

    assert result["available"] is False
    assert result["is_match"] is False
    assert "최소 80거래일" in result["reason"]


def test_snapshot_exposes_divergence_contract():
    closes, volumes = _series(recent_volume=350.0, latest_price=99.0)
    highs = [price * 1.01 for price in closes]
    lows = [price * 0.99 for price in closes]
    opens = list(closes)

    snapshot = build_snapshot_from_ohlcv(
        "TEST", closes, highs, lows, volumes, opens=opens,
    )

    signal = snapshot.volume_price_divergence
    assert signal["is_match"] is True
    assert signal["recent_days"] == VPD_RECENT_DAYS
    assert signal["baseline_days"] == VPD_BASELINE_DAYS
    assert signal["min_volume_ratio"] == VPD_MIN_VOLUME_RATIO
    assert signal["max_abs_price_change_pct"] == VPD_MAX_ABS_PRICE_CHANGE_PCT
