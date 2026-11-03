# Attribution

## StockFlow
- Source: https://github.com/nizhanthhhh/stockflow.git
- Accessed: 2026-08-29
- License: No LICENSE file found at time of integration (default copyright). This integration is a
  clean-room reimplementation of ideas (feature schema 34 cols, 14-day ±3% dead-zone label, time-based split,
  LightGBM + Platt scaling) with no verbatim copy of StockFlow source blocks >10 lines.
  Feature formulas (RSI, MACD, etc.) are public domain. The StockFlow pretrained model file (`lgbm_model.pkl`)
  was not copied; a new model was trained on synthetic data via `scripts/train_ml_model.py`.
- If you are the StockFlow author and wish to clarify license, please open an issue.

## Other
- HybridTurtle v6.0, k-ant-daily, etc. as per existing StockOracle `market_briefing/__init__.py` headers.
