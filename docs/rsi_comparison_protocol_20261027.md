# RSI / Laguerre RSI protocol (frozen before results)

Purpose: determine whether replacing the current oscillator is supported, not maximize a displayed backtest return.

- Universe: all 52 existing `datasets/audit_cache/*.csv`; historical cached adjusted daily OHLCV, not current quotes. Record SHA256 and date range of each input.
- Invalid OHLC rows: repair only relative envelope noise <=1e-6 with closes unchanged. Split at genuinely invalid rows, preserve dates, reset recursive indicators and warmup on each independent segment. No trade spans a bad row.
- Common eligibility: 126 valid consecutive bars before a candidate entry. Minimum segment 150 bars. No selecting tickers by performance.
- Train: 2022-01-01 <= entry and exit <2024-01-01. Validation: 2024-01-01 to 2025-01-01. Test: 2025-01-01 through the cache end, before 2026-10-27. No train outcome crosses its endpoint.
- Laguerre grid: gamma .3, .5, .7, default .5. First-price initialization, flat denominator ->50, output 0..100. No adaptive gamma or threshold search.
- Track 1: standalone upward recovery from oversold. Current Wilder-style RSI14 cross back above30 versus Laguerre cross back above20 (native conventional rule). Also evaluate matched30 thresholds as sensitivity; do not select it using test results.
- Track 2: existing dynamic RSI bands, confirmed divergence and50 recovery, unchanged config. Substitute only oscillator input. Baseline uses the existing dynamic module's native RSI.
- Isolated entry experiment: same next-open execution, long only, ATR14 rolling mean, signal-close -2ATR stop and +4ATR target, max20 sessions, one position per symbol/variant. Conservative same-bar stop priority, stop gaps at open, no favourable target gap fill, exclude unfinished trades. These are controlled entry experiments, not a full service or native dynamic-exit replay.
- Base costs: KRX .20%, US .10% round trip; slippage .10% each side. Stress: +.30 percentage points round-trip cost. Costs are research assumptions, not current legal/broker fee quotations.
- Select gamma by training-only equal-symbol mean log equity, separately for each track. Keep all tickers including zero-trade symbols in equal-symbol comparisons.
- Adoption: selected Laguerre must improve equal-symbol returns on validation and test in BOTH markets, final symbol-paired bootstrap95% lower bound >0 in both markets (seed20261027,2000 resamples), at least30 final trades per market, final profit factor >1 and positive mean trade return, no worse mean per-symbol completed-trade equity drawdown, and maintain test superiority under stressed cost. Both primary tracks must pass before any general replacement.
- Report all grid results and exclusions. No test-set retuning. Shared historical caches have been used in earlier project research: this is a new comparison holdout, not globally untouched research data. Survivor selection, daily-bar path ambiguity, segment resets and correlated symbols remain limitations. Bootstrap intervals are descriptive and do not correct multiple model selection or all market dependence.

Sources to review: Ehlers (2004), *Cybernetic Analysis for Stocks and Futures*, chapter14; Backtrader's official Laguerre RSI source; TeleTrader's official indicator documentation. Reimplement the mathematical recurrence; do not copy third-party source code.
