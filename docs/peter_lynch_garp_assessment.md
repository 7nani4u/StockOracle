# Peter Lynch-Style GARP Assessment

## Source Review

The supplied Naver Premium article, ["'월가 영웅' 피터 린치 스타일 10선"](https://contents.premium.naver.com/itooza/snowball/contents/250429180855206sr) (2025-04-29), defines a five-condition US stock screen:

1. `PER < 25`
2. `debt-to-equity < 80%`
3. Three-year EPS CAGR between `20%` and `50%`
4. `PEG < 1.5`
5. Market capitalization below `$10B` (approximately KRW 14 trillion in the article)

The broader investment framework is GARP, balancing growth with valuation rather than treating growth or a technical signal as sufficient. Peter Lynch's *One Up On Wall Street* and *Beating the Street* are the primary descriptions of his approach. The public biographical overview also identifies GARP and "invest in what you know" as central ideas: [Peter Lynch, Wikipedia](https://en.wikipedia.org/wiki/Peter_Lynch). These sources are context only; the implementation intentionally follows the article's explicit, reproducible thresholds rather than inventing additional Lynch rules.

## Implementation

`market_briefing/peter_lynch.py` returns a `peter_lynch` object from `/api/stock`.

- It checks all five conditions independently and returns `pass`, `fail`, or `unavailable` for each one.
- It only sets `eligible: true` when all five conditions pass. Missing values never pass a criterion.
- It calculates three-year EPS CAGR from four consecutive annual reported EPS values. Quarterly EPS, TTM growth, and analyst forecasts are not substituted.
- It calculates `PEG = PER / EPS CAGR(%)` from that exact reported CAGR. Provider-supplied PEG is not used because its growth horizon and forward/TTM definition may differ.
- It uses the article's `$10B` threshold for US stocks and its stated approximate KRW 14 trillion threshold for KRX stocks.

The result appears in the existing fundamental diagnosis panel. It is deliberately not blended into the short-horizon price probability: a long-term fundamental filter does not validate a 1- to 14-day technical forecast.

## Data and Limits

- Primary live inputs are Yahoo Finance via `yfinance`; KRX PER and market capitalization can fall back to Naver Finance. These are data-provider fields, not audited filings.
- For an investment decision, US values should be checked against the issuer's SEC filings and KRX values against DART filings. The official sources are [SEC EDGAR](https://www.sec.gov/edgar/search/) and [DART](https://dart.fss.or.kr/).
- Debt-to-equity is less comparable for banks, insurers, and other financial firms. The response adds a warning for those sectors instead of silently treating the ratio as equivalent to an industrial company.
- EPS CAGR is undefined across negative or zero start/end EPS. Such cases are returned as `unavailable`, not as artificial high growth.
- Passing a screen does not imply data recency, earnings quality, competitive durability, liquidity, or future returns. It is a research filter, not investment advice or an automated order signal.
