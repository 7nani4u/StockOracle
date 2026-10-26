"""Execution regressions: measured outcomes must use attainable fill prices."""
import pandas as pd
import pytest

from scripts.audit_box_tactics import plan_event, support_event, breakout_event, load_prices, Bucket
from scripts import audit_box_tactics as audit


def prices(rows):
    return pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close'],
                        index=pd.bdate_range('2025-01-01', periods=len(rows))).assign(Volume=1000)


def tactics(targets=(110, 120)):
    return {'plan': {'status': 'ready', 'side': 'long',
                     'entry': {'mid': 100, 'high': 105}, 'stop': {'price': 90},
                     'targets': [{'price': p, 'pct': 50} for p in targets]},
            'midline': {'state': 'normal', 'price': 110}}


def test_next_open_fill_defines_realized_risk_and_reward():
    frame = prices([(100, 101, 99, 100), (95, 121, 94, 115)])
    result = plan_event(frame, 0, tactics(), 1)
    assert result['r'] == pytest.approx(4.0)  # half at 110, half at 120; risk=95-90
    assert result['fill_price'] == 95


def test_stop_gap_exits_at_open_instead_of_unattainable_stop():
    frame = prices([(100, 101, 99, 100), (100, 105, 95, 100), (80, 85, 75, 80)])
    result = plan_event(frame, 0, tactics(), 2)
    assert result['r'] == -2.0
    assert result['exit'] == 'stop_gap'


def test_one_target_closes_entire_position():
    plan = tactics((110,))
    plan['plan']['targets'][0]['pct'] = 100
    frame = prices([(100, 101, 99, 100), (100, 115, 99, 101)])
    assert plan_event(frame, 0, plan, 1)['r'] == 1.0


def test_partial_exit_then_gap_accounts_for_remaining_half():
    frame = prices([(100, 101, 99, 100), (100, 111, 99, 108), (80, 85, 75, 80)])
    assert plan_event(frame, 0, tactics(), 2)['r'] == -0.5


def test_unfinished_trade_is_not_reported_as_realized_horizon_exit():
    frame = prices([(100, 101, 99, 100), (100, 105, 95, 102)])
    assert plan_event(frame, 0, tactics(), 10) is None


def test_cost_and_slippage_reduce_result():
    frame = prices([(100, 101, 99, 100), (100, 121, 99, 115)])
    zero = plan_event(frame, 0, tactics(), 1)
    stressed = plan_event(frame, 0, tactics(), 1, cost_pct=0.2, slippage_pct=0.1)
    assert stressed['r'] < zero['r']
    assert stressed['net_return_pct'] < zero['net_return_pct']


def test_same_bar_stop_has_priority():
    frame = prices([(100, 101, 99, 100), (100, 121, 89, 115)])
    assert plan_event(frame, 0, tactics(), 1)['r'] == -1.0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1])
def test_invalid_future_bar_is_rejected(value):
    frame = prices([(100, 101, 99, 100), (100, value, 95, 102)])
    assert plan_event(frame, 0, tactics(), 1) is None


def test_support_gap_to_target_resolves_before_later_stop():
    frame = prices([(100, 101, 99, 100), (115, 116, 80, 90)])
    box = {'support': {'low': 99, 'high': 101}, 'atr': 1}
    result = support_event(frame, 0, box, tactics(), 1)
    assert result['outcome'] == 'midline_first'
    assert result['ambiguous'] is False


def test_unresolved_support_is_kept_in_denominator():
    frame = prices([(100, 101, 99, 100), (100, 105, 99, 102)])
    result = support_event(frame, 0, {'support': {'low': 99, 'high': 101}, 'atr': 1}, tactics(), 1)
    assert result['outcome'] == 'unresolved'


def test_exact_three_bar_breakout_horizon_is_sufficient():
    frame = prices([(100, 101, 99, 100), (102, 103, 101, 102)] + [(103, 104, 102, 103)] * 3)
    result = breakout_event(frame, 1, {'resistance': {'low': 100, 'high': 101}, 'atr': 1}, tactics())
    assert result['held'] is True


def test_incomplete_cache_schema_is_skipped(tmp_path):
    path = tmp_path / 'bad.csv'
    pd.DataFrame({'Close': [100]}, index=['2025-01-01']).to_csv(path)
    assert load_prices(str(path)) is None


def test_unresolved_events_are_in_primary_rate_denominator():
    bucket = Bucket()
    bucket.add(True, False)
    bucket.add(False, True)
    bucket.add(False, False)
    assert bucket.as_dict()['midline_rate'] == .3333
    assert bucket.as_dict()['resolved_only_rate'] == .5


def test_post_target_stop_move_applies_next_bar():
    plan = tactics()
    plan['plan']['stop_after_t1'] = 100
    frame = prices([(100, 101, 99, 100), (100, 111, 95, 105), (105, 106, 99, 101)])
    result = plan_event(frame, 0, plan, 2)
    assert result['r'] == .5
    assert result['exit_index'] == 2


def test_profit_gap_is_resolved_before_intraday_low():
    frame = prices([(100, 101, 99, 100), (100, 105, 95, 100), (125, 126, 80, 90)])
    assert plan_event(frame, 0, tactics(), 2)['r'] == 1.5


@pytest.mark.parametrize('weights', [(50, 20), (100, 100), (-1, 101)])
def test_invalid_target_weights_are_rejected(weights):
    plan = tactics()
    for t, w in zip(plan['plan']['targets'], weights):
        t['pct'] = w
    frame = prices([(100, 101, 99, 100), (100, 121, 99, 115)])
    assert plan_event(frame, 0, plan, 1) is None


def test_audit_rejects_overlapping_positions_and_propagates_costs(monkeypatch):
    frame = prices([(100, 101, 99, 100)] * 115)
    monkeypatch.setattr(audit, 'analyze_box_range', lambda dd: {'state': 'range'})
    monkeypatch.setattr(audit, 'analyze_trend_boxes', lambda dd: {})
    monkeypatch.setattr(audit, 'analyze_box_tactics', lambda *a, **k: {})
    monkeypatch.setattr(audit, 'support_event', lambda *a: None)
    monkeypatch.setattr(audit, 'breakout_event', lambda *a: None)
    def event(frame, i, tactics, horizon, **kwargs):
        assert kwargs == {'cost_pct': .2, 'slippage_pct': .1}
        return {'r': 1, 'exit': 'horizon', 'entry_index': i+1, 'exit_index': i+10}
    monkeypatch.setattr(audit, 'plan_event', event)
    result = audit.audit_ticker('TEST', frame, 1, cost_pct=.2, slippage_pct=.1)
    assert len(result['plans']) == 2  # entries 91 and 101; no simultaneous positions


def test_cli_skips_invalid_cache_and_accepts_plain_output_filename(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    cache.mkdir()
    pd.DataFrame({'Close': [100]}).to_csv(cache / 'BROKEN.csv')
    prices([(100, 101, 99, 100)] * 200).to_csv(cache / 'GOOD.csv')
    monkeypatch.setattr(audit, 'CACHE_DIR', str(cache))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit, 'audit_ticker', lambda *a, **k:
        {'support': [], 'breakout': [], 'plans': [], 'states': audit.Counter(), 'regimes': audit.Counter()})
    assert audit.main(['--out', 'result.json', '--cost-pct', '.2', '--slippage-pct', '.1']) == 0
    import json
    report = json.loads((tmp_path / 'result.json').read_text(encoding='utf-8'))
    assert report['tickers'] == 1
    assert report['execution']['cost_round_trip_pct'] == .2
