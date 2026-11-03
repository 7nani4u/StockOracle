import sys
sys.path.insert(0,'.')
from tests.test_buy_entry_steps import _sample_buy_dd, _calculate
from api.index import _market_tick_size, _round_market_price
import math
dd=_sample_buy_dd()
# run calc and also manually trace
result=_calculate(dd)
band=result['aggressive_bands'][0]
print("band range", band['range'])
# also call internal function to see boundaries
# Let's monkey patch to print
from api.index import calc_buy_price
import api.index as idx
orig_build = idx.calc_buy_price
# Instead directly compute with same logic but print
# We'll just print the steps we got
for s in band['steps']:
    print(s['label'], s['price'], s['price_range'])

# Now test tick
print("tick for 113.515", _market_tick_size(113.515, "KRX"))
print("ceil 113.515", _round_market_price(113.515, "KRX", "ceil"))
print("floor 113.515", _round_market_price(113.515, "KRX", "floor"))
print("ceil 112.545", _round_market_price(112.545, "KRX", "ceil"))
print("floor 112.545", _round_market_price(112.545, "KRX", "floor"))
