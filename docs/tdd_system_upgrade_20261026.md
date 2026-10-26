# TDD evidence — StockOracle 2026-10-26

## Scope and journeys

Derived from the user's project-wide improvement request, supplied video, and subsequent forecast UI removal request. Existing worktree changes were preserved. Detailed structure, analysis, ten improvement proposals, PDCA, video timestamps and APA sources are in `system_upgrade_20261026.md`.

1. A user sees one final risk decision across box entry cards, execution status and pyramiding.
2. A researcher measures trades at actual next-open fills with gaps, partial exits, explicit costs and no overlapping positions.
3. A user receives coherent directional probabilities, explicit neutral mass and approved model provenance.
4. Requested forecast sections disappear without disabling the remaining forecast renderer.

## RED / GREEN evidence

| Behavior | Test target | RED observed | GREEN observed |
|---|---|---|---|
| Fill-based R, stop gaps, partial/single exits, censoring, invalid bars, support gap order, unresolved support, exact breakout horizon, broken cache schema | tests/test_box_audit_execution.py | 13 failed, 1 passed | 14 passed after initial implementation; expanded to 22 passed |
| Costs propagated to ticker replay; overlapping positions excluded; CLI plain filename and corrupt cache | same target | 2 failed, 20 passed | 22 passed |
| Final gates dominate breakout candidate; raw plan and pyramid agree; weighted reward; no invented 100-based targets | tests/test_box_risk_gate_20261026.py | 16 failed, 1 passed | 17 passed; related box targets 53 passed |
| Final execution summary also says wait | same target | 5 failed, 12 passed | included in subsequent passing run |
| Missing/malformed approval, complement probability, neutral-dominant action, model class ordering | tests/test_ml_probability_contract_20261026.py | 10 failed, 1 passed | 11 passed initially; expanded to 43 passed |
| Neutral model abstains in both blend and correlation consumers | tests/test_ml_neutral_consumers_20261026.py | 2 failed | focused consumers + chain regression completed successfully |
| Removed sections, retained wave analysis and valid DOM references | tests/test_forecast_ui_cleanup_20261026.py + signal confidence layout + prediction outlook | 3 failed, 63 passed | 66 passed |
| Runtime renderer no longer expects deleted monthly summary | tests/test_forecast_integrity.py | 1 failure in entire suite, 693 passed | runtime target + box gates 22 passed |

Commands actually run:

```powershell
python -m pytest tests -q --disable-warnings --maxfail=3
python -m pytest tests/test_box_audit_execution.py -q --tb=short
python -m pytest tests/test_ml_probability_contract_20261026.py tests/test_ml_evaluation_integrity.py tests/test_ml_feature_parity.py tests/test_training_audit_fixes.py -q --disable-warnings
python -m coverage run --timid --data-file=.coverage.focused20261026 --source=market_briefing.box_prediction,market_briefing.box_tactics,market_briefing.ml_predictor,scripts.audit_box_tactics -m pytest tests/test_box_audit_execution.py tests/test_box_risk_gate_20261026.py tests/test_box_prediction.py tests/test_box_tactics.py tests/test_box_tactics_hardening.py tests/test_ml_probability_contract_20261026.py tests/test_ml_evaluation_integrity.py tests/test_ml_feature_parity.py tests/test_training_audit_fixes.py -q --disable-warnings
python scripts/audit_box_tactics.py --offline --stride 3 --cost-pct .2 --slippage-pct .1 --out docs/backtests/box_tactics_audit_20261026_costs.json
```

Baseline: **643 passed, 6 subtests passed**. Final suite: **728 passed, 6 subtests passed in 24.87s**. Neutral consumers and prediction-chain target: **41 passed**.

## Coverage and limits

Focused coverage command: **139 passed**. Statement coverage: box_prediction **95%**, box_tactics **90%**, audit_box_tactics **83%**, ml_predictor entire module **51%**, combined four modules **77%**. This is not whole-project 80% coverage. Network/index-fetch paths in ml_predictor were outside the targeted change. The changed ML functions measured independently at load_model **90.3%**, probability details **100%**, compatibility probability API **100%**, predict_from_ohlcv **88.9%**. Coverage JSONs are in `tmp/upgrade_coverage_20261026.json` and `tmp/ml_contract_coverage_20261026.json`.

Coverage's default tracer on the full Python3.14 suite produced NumPy `_NoValueType` errors (363 passed, 3 failed before stopping), whereas normal pytest passed those tests. Focused coverage was rerun with the timid tracer; this tooling limitation is not reported as a successful full coverage run. Runtime JavaScript is exercised by the existing Node VM renderer tests; a full real-browser end-to-end external-data test was not run.

## Research evidence

The corrected raw-rule audit reports **52 tickers, 19,700 snapshots**. Support: **284 events**, 108 midline-first, 155 stop-first, 21 unresolved; all-event midline rate **38.03%**, resolved-only **41.06%**. Strong-magnet rate **29.41%**, weak **48.28%**, so this sample does not establish the video's claimed strong-magnet advantage.

With illustrative .2% round-trip cost and .1% each-side slippage: **15 completed plans**, mean **0.7391R**, win rate **40%**, descriptive Wilson interval **19.82–64.25%**. These are selected historical raw rules, not portfolio return or service accuracy. Entry at better opening prices can increase R even after costs. Definition changes and sample exclusions prevent a like-for-like performance claim against the previous report. Final service trend/quote/overall-risk gates and pyramiding are explicitly excluded. Correlated observations, survivor selection, small sample and daily intrabar ambiguity remain.

## Checkpoints and existing changes

RED test checkpoints: `6722e06`, `ae25aa5`, on the active main branch. Only new test files were committed in those checkpoints; pre-existing staged deletions and unrelated changes were not included. Production files already contained substantial uncommitted user work (some entirely untracked). GREEN production changes remain reviewable in the workspace instead of bundling that earlier work into a large incidental commit. Thus full production-stage commit checkpoints from the skill are not claimed; the RED/GREEN evidence is preserved here.

## Self-evaluation

Accuracy 4/5: reproducible defects fixed, historical outcomes and probability meanings separated; live provider results not verified. Completeness 3/5: requested UI changes and identified correctness fixes completed; whole-repository 80% coverage, complete browser E2E, independent strategy profitability and new model provenance are not established. Clarity 4/5: full structure and timestamps documented, with a concise user-facing outcome. Actionability 4/5: local code, tests and reproducible audit command available; deployment and model provenance recovery remain separate work. Conciseness 4/5: detailed requested analysis lives in the report. Average **3.8/5**, a self-assessment rather than an external certification. Highest priorities: recover trustworthy training provenance, replay point-in-time final service responses, and evaluate untouched periods with realistic costs.
