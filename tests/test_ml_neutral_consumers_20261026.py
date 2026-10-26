from api.index import _ml_blend_weight
from market_briefing.correlation_engine import correlate_and_narrow


def test_neutral_action_has_no_blend_weight():
    assert _ml_blend_weight(.65, True, direction='NEUTRAL') == 0
    assert _ml_blend_weight(.65, True, direction='UP') == .25


def _ml_vote(prediction):
    result = correlate_and_narrow(
        symbol='TEST', market='US', dd={'Close': [100] * 80}, last_price=100,
        atr=2, score=50, prob_up_base=50, prob_down_base=50,
        target_price={'min_price': 105, 'max_price': 110}, signal_confidence=None,
        indicator_signals={}, candlestick_patterns=[], pullback_analysis=None, investor_flow=None,
        ml_prediction=prediction,
        regime='NEUTRAL', pct_change=0, volume_ratio=1, candle_up=True,
        rsi=50, macd_gap=0, event_risk=None)
    return result['correlation']['dimensions']['ml']


def test_correlation_does_not_turn_conditional_odds_into_neutral_model_vote():
    assert _ml_vote({'direction': 'NEUTRAL', 'prob_up': .9, 'prob_neutral': .9, 'trust': 1}) == 0


def test_heuristic_fallback_is_not_an_independent_ml_vote():
    assert _ml_vote({'direction': 'UP', 'prob_up': .68, 'fallback': True}) == 0


def test_valid_directional_model_still_votes():
    assert _ml_vote({'direction': 'UP', 'prob_up': .68, 'trust': 1}) > 0
