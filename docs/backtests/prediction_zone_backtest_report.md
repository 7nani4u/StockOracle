# Prediction Zone Backtest Report

Generated: 2026-06-11

## Universe

KRX 100:
005930.KS, 000660.KS, 373220.KS, 207940.KS, 005380.KS, 000270.KS, 068270.KS, 105560.KS, 055550.KS, 005490.KS, 035420.KS, 035720.KS, 051910.KS, 006400.KS, 012330.KS, 028260.KS, 012450.KS, 034020.KS, 247540.KQ, 196170.KQ, 009150.KS, 009540.KS, 015760.KS, 033780.KS, 086790.KS, 003670.KS, 259960.KS, 086520.KQ, 028300.KQ, 145020.KQ, 263750.KQ, 293490.KQ, 112040.KQ, 041510.KQ, 011200.KS, 010140.KS, 058470.KQ, 035900.KQ, 214150.KQ, 357780.KQ, 095340.KQ, 140860.KQ, 098460.KQ, 222800.KQ, 240810.KQ, 178320.KQ, 032830.KS, 066570.KS, 000810.KS, 316140.KS, 018260.KS, 096770.KS, 090430.KS, 011070.KS, 086280.KS, 251270.KS, 034730.KS, 003550.KS, 010130.KS, 024110.KS, 030200.KS, 017670.KS, 352820.KS, 326030.KS, 138040.KS, 000100.KS, 018880.KS, 267260.KS, 047810.KS, 071050.KS, 004020.KS, 005830.KS, 000720.KS, 010950.KS, 161390.KS, 021240.KS, 307950.KS, 036570.KS, 128940.KS, 271560.KS, 039490.KS, 078930.KS, 008770.KS, 180640.KS, 006800.KS, 071320.KS, 005940.KS, 272210.KS, 004990.KS, 023530.KS, 383220.KS, 402340.KS, 403870.KS, 278470.KS, 348370.KQ, 277810.KQ, 039030.KQ, 067310.KQ, 253450.KQ, 091990.KQ.

US 100:
AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, BRK-B, LLY, AVGO, JPM, V, UNH, XOM, MA, ORCL, COST, NFLX, JNJ, WMT, PG, BAC, HD, CVX, MRK, ABBV, CRM, KO, AMD, ACN, PEP, TMO, LIN, MCD, CSCO, PM, DIS, TXN, ABT, ISRG, DHR, AMGN, NEE, WFC, RTX, CAT, VZ, CMCSA, INTU, ADBE, IBM, GS, MS, BKNG, HON, SPGI, QCOM, NOW, UNP, ETN, LOW, T, GE, AXP, SYK, BLK, MDT, GILD, ELV, DE, ADI, PLD, CI, MMC, VRTX, CB, SO, COP, SBUX, PANW, MO, APD, EOG, BSX, LRCX, TT, ADP, ITW, ANET, REGN, PGR, KLAC, ZTS, CME, ICE, ECL, HUM, MCO, PSA, NOC.

Yahoo Finance returned no usable 3-year daily data for 091990.KQ and MMC during this run, so effective measured symbols were 99 KRX and 99 US tickers.

## Method

- Data: 3-year daily OHLCV from Yahoo Finance.
- Indicators: StockOracle `add_indicators()` output.
- Signals: rolling samples every 12 trading days over the recent backtest window.
- Forward window: 20 trading days after each signal.
- Metrics: band entry rate, bounce success, stop hit, extra drop after entry, average return, max drawdown, ATR%, volume ratio.

## Accuracy Comparison

| Market | Zone | Baseline entry | New entry | Baseline bounce | New bounce | Baseline stop | New stop | Baseline extra drop | New extra drop | New avg MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KRX | 1차 | 88.96% | 68.35% | 73.20% | 77.44% | 57.31% | 56.45% | 75.62% | 77.24% | -9.107% |
| KRX | 2차 | 75.96% | 49.76% | 63.03% | 70.64% | 47.34% | 43.71% | 75.80% | 72.67% | -8.724% |
| US | 1차 | 90.30% | 72.73% | 70.92% | 74.63% | 63.98% | 61.30% | 79.94% | 80.09% | -6.203% |
| US | 2차 | 81.08% | 53.80% | 59.47% | 63.45% | 53.32% | 50.56% | 78.82% | 76.60% | -6.082% |

## Failure Notes

- High ATR names still show large post-entry drawdowns; shallow primary entries remain fragile in volatility shock.
- Low-volume weakness under MA20 was under-penalized, so it is now added to downside risk.
- KRX high-volatility thresholds need to be stricter than US thresholds because the tested KRX median ATR% was materially higher.

Worst current-run examples:
- KRX: 348370.KQ had repeated severe drawdowns after entry, especially with ATR above 6%.
- US: HUM, ZTS, INTU showed large drawdowns despite some signals not reaching severe risk before the event.

## Implemented Changes

- Added market-specific ATR% risk penalties.
- Added low-volume plus MA20-break penalty.
- Lowered hold thresholds from 72/55/35 to 68/50/30 risk points.
- Increased base ATR depth for 1차 bands by market.
- Made 2차 bands deeper and more selective.
- Kept risk-card allocation and max-loss controls visible.

