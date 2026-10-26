import pytest

from market_briefing.box_tactics import build_plan


def plan(rows, current=100.5, pressure='buy'):
    return build_plan({'key': 'breakout_confirmed'},
        {'resistance': {'low': 100, 'high': 101}}, {}, {}, {}, {'verdict': pressure},
        current, 1.0, rows=rows, volume_context={'ratio': 1.5})


@pytest.mark.parametrize('rows', [None, [], [{'low': 103, 'close': 104}],
                                  [{'low': 99, 'close': 99.5}]])
def test_quote_near_resistance_does_not_replace_confirmed_retest(rows):
    out = plan(rows)
    assert out['status'] == 'wait'
    assert out['tradeable'] is False
    assert out['entry_checks']


def test_completed_retest_with_buy_pressure_can_be_ready():
    out = plan([{'low': 100.3, 'close': 101.1}])
    assert out['status'] == 'ready'
    assert all(c['ok'] for c in out['entry_checks'])


def test_retest_with_sell_pressure_cannot_authorize_entry():
    assert plan([{'low': 100.3, 'close': 101.1}], pressure='sell')['status'] == 'wait'


@pytest.mark.parametrize('quote', [None, 110])
def test_pyramid_cannot_be_ready_when_initial_entry_is_not_ready(quote):
    out = plan([{'low': 100.3, 'close': 101.1}], current=quote)
    assert out['status'] == 'wait'
    assert out['pyramid']['ready'] is False
