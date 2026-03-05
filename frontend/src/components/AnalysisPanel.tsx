'use client';

import { useEffect, useRef, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// 타입 정의
// ─────────────────────────────────────────────────────────────────────────────
interface StockData {
  symbol: string;
  company: string;
  market: string;
  last_close: number;
  prev_close: number;
  pct_change: number;
  rsi: number;
  volume: number;
  atr: number;
  score: number;
  analysis_steps: AnalysisStep[];
  candlestick_patterns: CandlePattern[];
  chart_data: ChartData;
  forecast: ForecastData | null;
  xgb_forecast: number[] | null;
  risk_scenarios: RiskScenarios;
  news: NewsItem[];
  naver: NaverData | null;
}

interface AnalysisStep {
  step: string;
  result: string;
  score: number;
}

interface CandlePattern {
  name: string;
  desc: string;
  direction: '상승' | '하락' | '중립';
  conf: number;
  impact: string;
}

interface ChartData {
  dates: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
  ma20: (number | null)[];
  ma60: (number | null)[];
  bb_upper: (number | null)[];
  bb_lower: (number | null)[];
  rsi: (number | null)[];
  macd: (number | null)[];
  signal_line: (number | null)[];
}

interface ForecastData {
  dates: string[];
  yhat: number[];
  yhat_upper: number[];
  yhat_lower: number[];
}

interface RiskScenarios {
  conservative: RiskScenario;
  balanced: RiskScenario;
  aggressive: RiskScenario;
}

interface RiskScenario {
  label: string;
  target: number;
  stop: number;
  ratio: string;
  desc: string;
  icon: string;
}

interface NewsItem {
  title: string;
  link: string;
  publisher: string;
  published: string;
}

interface NaverData {
  price: string | null;
  market_cap: string | null;
  per: string | null;
  pbr: string | null;
  opinion: string | null;
  news: { title: string; link: string }[];
  disclosures: { title: string; link: string }[];
}

interface Props {
  data: StockData | null;
  loading: boolean;
  error: string;
  market: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// 유틸리티
// ─────────────────────────────────────────────────────────────────────────────
const fmt = (v: number, isKrx: boolean) =>
  isKrx ? `${v.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}원`
         : `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const scoreColor = (s: number) =>
  s >= 70 ? 'text-green-400' : s >= 40 ? 'text-yellow-400' : 'text-red-400';

const scoreBarColor = (s: number) =>
  s >= 70 ? 'bg-green-500' : s >= 40 ? 'bg-yellow-500' : 'bg-red-500';

// ─────────────────────────────────────────────────────────────────────────────
// SVG 캔들차트 (lightweight-charts 대신 순수 SVG, SSR safe)
// ─────────────────────────────────────────────────────────────────────────────
function CandleChart({
  chartData, market, forecast, xgbForecast,
}: {
  chartData: ChartData;
  market: string;
  forecast: ForecastData | null;
  xgbForecast: number[] | null;
}) {
  const [activeTab, setActiveTab] = useState<'price' | 'rsi' | 'macd' | 'forecast'>('price');
  const isKrx = market === 'KRX';

  const n = chartData.dates.length;
  if (n < 2) return <div className="text-gray-500 text-sm p-4">데이터 부족</div>;

  // 최근 120개만 표시
  const slice = Math.min(n, 120);
  const dates = chartData.dates.slice(-slice);
  const opens = chartData.open.slice(-slice);
  const highs = chartData.high.slice(-slice);
  const lows = chartData.low.slice(-slice);
  const closes = chartData.close.slice(-slice);
  const volumes = chartData.volume.slice(-slice);
  const ma20 = chartData.ma20.slice(-slice);
  const ma60 = chartData.ma60.slice(-slice);
  const bbU = chartData.bb_upper.slice(-slice);
  const bbL = chartData.bb_lower.slice(-slice);
  const rsiArr = chartData.rsi.slice(-slice);
  const macdArr = chartData.macd.slice(-slice);
  const signalArr = chartData.signal_line.slice(-slice);

  const W = 900, H = 360, VPAD = 20, HPAD = 50;
  const chartW = W - HPAD;
  const chartH = H - VPAD * 2;

  const allPrices = [...highs, ...lows, ...bbU, ...bbL].filter(v => v != null) as number[];
  const minP = Math.min(...allPrices);
  const maxP = Math.max(...allPrices);
  const rangeP = maxP - minP || 1;

  const scaleY = (v: number) => VPAD + chartH * (1 - (v - minP) / rangeP);
  const scaleX = (i: number) => HPAD + (i / (slice - 1)) * chartW;

  const candleW = Math.max(2, (chartW / slice) * 0.6);

  // MA 선 path
  const linePath = (arr: (number | null)[]) => {
    let d = '';
    arr.forEach((v, i) => {
      if (v == null) return;
      const x = scaleX(i), y = scaleY(v);
      d += d ? `L${x},${y}` : `M${x},${y}`;
    });
    return d;
  };

  // 볼린저 밴드 fill
  const bbFill = () => {
    let top = '', bot = '';
    bbU.forEach((u, i) => {
      if (u == null) return;
      const x = scaleX(i), y = scaleY(u);
      top += top ? `L${x},${y}` : `M${x},${y}`;
    });
    const revBbL = [...bbL].reverse();
    revBbL.forEach((l, ri) => {
      if (l == null) return;
      const i = slice - 1 - ri;
      const x = scaleX(i), y = scaleY(l);
      bot += `L${x},${y}`;
    });
    return top + bot + 'Z';
  };

  // 거래량 차트
  const maxVol = Math.max(...volumes.filter(v => v > 0));
  const volH = 60;
  const volTop = H + 10;

  return (
    <div>
      {/* 탭 */}
      <div className="flex gap-2 mb-3">
        {(['price', 'rsi', 'macd', 'forecast'] as const).map(t => (
          <button
            key={t}
            className={`tab-btn ${activeTab === t ? 'active' : ''}`}
            onClick={() => setActiveTab(t)}
          >
            {t === 'price' ? '📊 가격/지표' : t === 'rsi' ? '📉 RSI' : t === 'macd' ? '📈 MACD' : '🔮 예측'}
          </button>
        ))}
      </div>

      {activeTab === 'price' && (
        <div className="overflow-x-auto">
          <svg viewBox={`0 0 ${W} ${H + volH + 20}`} className="w-full" style={{ minWidth: 600 }}>
            <defs>
              <linearGradient id="bbGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.08" />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.03" />
              </linearGradient>
            </defs>

            {/* 볼린저 밴드 fill */}
            <path d={bbFill()} fill="url(#bbGrad)" />

            {/* BB lines */}
            <path d={linePath(bbU)} stroke="#3b82f6" strokeWidth="0.8" fill="none" strokeDasharray="4 3" opacity="0.6" />
            <path d={linePath(bbL)} stroke="#3b82f6" strokeWidth="0.8" fill="none" strokeDasharray="4 3" opacity="0.6" />

            {/* MA lines */}
            <path d={linePath(ma20)} stroke="#f97316" strokeWidth="1.2" fill="none" />
            <path d={linePath(ma60)} stroke="#a78bfa" strokeWidth="1.2" fill="none" />

            {/* 캔들 */}
            {closes.map((c, i) => {
              const o = opens[i], h = highs[i], l = lows[i];
              const x = scaleX(i);
              const isBull = c >= o;
              const color = isKrx
                ? (isBull ? '#ef4444' : '#3b82f6')
                : (isBull ? '#22c55e' : '#ef4444');
              const bodyTop = scaleY(Math.max(o, c));
              const bodyBot = scaleY(Math.min(o, c));
              const bodyH = Math.max(1, bodyBot - bodyTop);
              return (
                <g key={i}>
                  <line x1={x} y1={scaleY(h)} x2={x} y2={scaleY(l)} stroke={color} strokeWidth="1" />
                  <rect x={x - candleW / 2} y={bodyTop} width={candleW} height={bodyH} fill={color} rx="0.5" />
                </g>
              );
            })}

            {/* 거래량 바 */}
            {volumes.map((v, i) => {
              const x = scaleX(i);
              const barH = maxVol > 0 ? (v / maxVol) * volH : 0;
              const isBull = closes[i] >= opens[i];
              const color = isKrx
                ? (isBull ? 'rgba(239,68,68,0.4)' : 'rgba(59,130,246,0.4)')
                : (isBull ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)');
              return (
                <rect
                  key={`v${i}`}
                  x={x - candleW / 2}
                  y={volTop + volH - barH}
                  width={candleW}
                  height={barH}
                  fill={color}
                  rx="0.5"
                />
              );
            })}

            {/* Y축 레이블 */}
            {[0, 0.25, 0.5, 0.75, 1].map(p => {
              const price = minP + rangeP * p;
              const y = scaleY(price);
              return (
                <g key={p}>
                  <line x1={HPAD} y1={y} x2={W} y2={y} stroke="#374151" strokeWidth="0.5" strokeDasharray="3 3" />
                  <text x={HPAD - 4} y={y + 4} fill="#6b7280" fontSize="9" textAnchor="end">
                    {isKrx ? price.toLocaleString('ko-KR', { maximumFractionDigits: 0 }) : price.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* X축 날짜 (10개) */}
            {Array.from({ length: 10 }, (_, k) => {
              const i = Math.floor((k / 9) * (slice - 1));
              return (
                <text key={k} x={scaleX(i)} y={H - 2} fill="#6b7280" fontSize="8" textAnchor="middle">
                  {dates[i]?.slice(5)}
                </text>
              );
            })}

            {/* 범례 */}
            <g>
              <rect x={HPAD + 5} y={VPAD + 5} width={8} height={3} fill="#f97316" rx="1" />
              <text x={HPAD + 15} y={VPAD + 9} fill="#f97316" fontSize="9">MA20</text>
              <rect x={HPAD + 55} y={VPAD + 5} width={8} height={3} fill="#a78bfa" rx="1" />
              <text x={HPAD + 65} y={VPAD + 9} fill="#a78bfa" fontSize="9">MA60</text>
              <rect x={HPAD + 105} y={VPAD + 5} width={8} height={2} fill="#3b82f6" rx="1" />
              <text x={HPAD + 115} y={VPAD + 9} fill="#3b82f6" fontSize="9">BB</text>
            </g>
          </svg>
        </div>
      )}

      {activeTab === 'rsi' && (
        <RSIChart rsiArr={rsiArr} dates={dates} />
      )}

      {activeTab === 'macd' && (
        <MACDChart macdArr={macdArr} signalArr={signalArr} dates={dates} />
      )}

      {activeTab === 'forecast' && (
        <ForecastChart
          chartData={chartData}
          forecast={forecast}
          xgbForecast={xgbForecast}
          isKrx={isKrx}
        />
      )}
    </div>
  );
}

function RSIChart({ rsiArr, dates }: { rsiArr: (number | null)[], dates: string[] }) {
  const W = 900, H = 200, VPAD = 20, HPAD = 50;
  const chartW = W - HPAD, chartH = H - VPAD * 2;
  const scaleY = (v: number) => VPAD + chartH * (1 - v / 100);
  const scaleX = (i: number) => HPAD + (i / (rsiArr.length - 1)) * chartW;

  let path = '';
  rsiArr.forEach((v, i) => {
    if (v == null) return;
    path += path ? `L${scaleX(i)},${scaleY(v)}` : `M${scaleX(i)},${scaleY(v)}`;
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ minWidth: 600 }}>
      {/* 기준선 */}
      {[30, 50, 70].map(v => (
        <g key={v}>
          <line x1={HPAD} y1={scaleY(v)} x2={W} y2={scaleY(v)}
            stroke={v === 70 ? '#ef4444' : v === 30 ? '#3b82f6' : '#374151'}
            strokeWidth="0.8" strokeDasharray="4 3" />
          <text x={HPAD - 4} y={scaleY(v) + 4} fill="#6b7280" fontSize="9" textAnchor="end">{v}</text>
        </g>
      ))}
      {/* 과매수/과매도 영역 fill */}
      <rect x={HPAD} y={VPAD} width={chartW} height={scaleY(70) - VPAD} fill="rgba(239,68,68,0.05)" />
      <rect x={HPAD} y={scaleY(30)} width={chartW} height={H - VPAD - scaleY(30)} fill="rgba(59,130,246,0.05)" />
      {/* RSI 라인 */}
      <path d={path} stroke="#facc15" strokeWidth="1.5" fill="none" />
      <text x={W / 2} y={VPAD - 5} fill="#facc15" fontSize="10" textAnchor="middle">RSI (14)</text>
    </svg>
  );
}

function MACDChart({ macdArr, signalArr, dates }: {
  macdArr: (number | null)[], signalArr: (number | null)[], dates: string[]
}) {
  const W = 900, H = 200, VPAD = 20, HPAD = 50;
  const chartW = W - HPAD, chartH = H - VPAD * 2;
  const vals = [...macdArr, ...signalArr].filter(v => v != null) as number[];
  const minV = Math.min(...vals, 0);
  const maxV = Math.max(...vals, 0);
  const range = maxV - minV || 1;
  const scaleY = (v: number) => VPAD + chartH * (1 - (v - minV) / range);
  const scaleX = (i: number) => HPAD + (i / (macdArr.length - 1)) * chartW;
  const zero = scaleY(0);

  let macdPath = '', sigPath = '';
  macdArr.forEach((v, i) => {
    if (v == null) return;
    macdPath += macdPath ? `L${scaleX(i)},${scaleY(v)}` : `M${scaleX(i)},${scaleY(v)}`;
  });
  signalArr.forEach((v, i) => {
    if (v == null) return;
    sigPath += sigPath ? `L${scaleX(i)},${scaleY(v)}` : `M${scaleX(i)},${scaleY(v)}`;
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ minWidth: 600 }}>
      <line x1={HPAD} y1={zero} x2={W} y2={zero} stroke="#374151" strokeWidth="0.8" strokeDasharray="3 3" />
      {/* MACD 히스토그램 */}
      {macdArr.map((v, i) => {
        if (v == null) return null;
        const x = scaleX(i);
        const barW = Math.max(1, chartW / macdArr.length * 0.6);
        const isBull = v >= 0;
        const barTop = isBull ? scaleY(v) : zero;
        const barH = Math.abs(scaleY(v) - zero);
        return (
          <rect key={i} x={x - barW / 2} y={barTop} width={barW} height={barH}
            fill={isBull ? 'rgba(34,197,94,0.5)' : 'rgba(239,68,68,0.5)'} rx="0.5" />
        );
      })}
      <path d={macdPath} stroke="#22c55e" strokeWidth="1.2" fill="none" />
      <path d={sigPath} stroke="#f97316" strokeWidth="1.2" fill="none" />
      {/* 범례 */}
      <rect x={HPAD + 5} y={VPAD} width={8} height={3} fill="#22c55e" rx="1" />
      <text x={HPAD + 15} y={VPAD + 4} fill="#22c55e" fontSize="9">MACD</text>
      <rect x={HPAD + 60} y={VPAD} width={8} height={3} fill="#f97316" rx="1" />
      <text x={HPAD + 70} y={VPAD + 4} fill="#f97316" fontSize="9">Signal</text>
    </svg>
  );
}

function ForecastChart({ chartData, forecast, xgbForecast, isKrx }: {
  chartData: ChartData;
  forecast: ForecastData | null;
  xgbForecast: number[] | null;
  isKrx: boolean;
}) {
  if (!forecast) {
    return <div className="text-gray-500 text-sm p-4 text-center">예측 데이터가 없습니다. 데이터가 더 필요합니다.</div>;
  }

  const W = 900, H = 320, VPAD = 20, HPAD = 50;
  const chartW = W - HPAD, chartH = H - VPAD * 2;

  const histClose = chartData.close.slice(-60);
  const histDates = chartData.dates.slice(-60);
  const fcDates = forecast.dates;
  const fcYhat = forecast.yhat;
  const fcUpper = forecast.yhat_upper;
  const fcLower = forecast.yhat_lower;

  const allVals = [...histClose, ...fcYhat, ...fcUpper, ...fcLower,
    ...(xgbForecast || [])].filter(v => v != null) as number[];
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const range = maxV - minV || 1;

  const totalDates = [...histDates, ...fcDates];
  const n = totalDates.length;
  const scaleX = (i: number) => HPAD + (i / (n - 1)) * chartW;
  const scaleY = (v: number) => VPAD + chartH * (1 - (v - minV) / range);

  // 히스토리 실제 가격 경로
  let histPath = '';
  histClose.forEach((v, i) => {
    const x = scaleX(i), y = scaleY(v);
    histPath += histPath ? `L${x},${y}` : `M${x},${y}`;
  });

  // 예측 경로 (히스토리 끝 지점부터)
  const fcOffset = histClose.length - 1;
  let fcPath = `M${scaleX(fcOffset)},${scaleY(histClose[histClose.length - 1])}`;
  fcYhat.forEach((v, i) => {
    fcPath += `L${scaleX(fcOffset + 1 + i)},${scaleY(v)}`;
  });

  // 신뢰구간 fill
  let bandPath = `M${scaleX(fcOffset + 1)},${scaleY(fcUpper[0])}`;
  fcUpper.forEach((v, i) => { bandPath += `L${scaleX(fcOffset + 1 + i)},${scaleY(v)}`; });
  [...fcLower].reverse().forEach((v, i) => {
    const idx = fcLower.length - 1 - i;
    bandPath += `L${scaleX(fcOffset + 1 + idx)},${scaleY(v)}`;
  });
  bandPath += 'Z';

  // XGBoost 예측 경로
  let xgbPath = '';
  if (xgbForecast) {
    xgbForecast.forEach((v, i) => {
      const x = scaleX(fcOffset + 1 + i), y = scaleY(v);
      xgbPath += xgbPath ? `L${x},${y}` : `M${scaleX(fcOffset)},${scaleY(histClose[histClose.length - 1])} L${x},${y}`;
    });
  }

  // 현재 시점 구분선
  const splitX = scaleX(fcOffset);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ minWidth: 600 }}>
      {/* 그리드 */}
      {[0, 0.25, 0.5, 0.75, 1].map(p => {
        const v = minV + range * p;
        const y = scaleY(v);
        return (
          <g key={p}>
            <line x1={HPAD} y1={y} x2={W} y2={y} stroke="#1f2937" strokeWidth="0.5" />
            <text x={HPAD - 4} y={y + 4} fill="#6b7280" fontSize="8" textAnchor="end">
              {isKrx ? v.toLocaleString('ko-KR', { maximumFractionDigits: 0 }) : v.toFixed(1)}
            </text>
          </g>
        );
      })}

      {/* 현재 구분선 */}
      <line x1={splitX} y1={VPAD} x2={splitX} y2={H - VPAD} stroke="#facc15" strokeWidth="1" strokeDasharray="5 3" />
      <text x={splitX} y={VPAD - 3} fill="#facc15" fontSize="8" textAnchor="middle">현재</text>

      {/* 신뢰구간 */}
      <path d={bandPath} fill="rgba(59,130,246,0.1)" />

      {/* 히스토리 */}
      <path d={histPath} stroke="#e5e7eb" strokeWidth="1.5" fill="none" />

      {/* HW 예측 */}
      <path d={fcPath} stroke="#3b82f6" strokeWidth="1.5" fill="none" strokeDasharray="6 3" />

      {/* XGBoost 예측 */}
      {xgbPath && <path d={xgbPath} stroke="#f97316" strokeWidth="1.5" fill="none" strokeDasharray="4 2" />}

      {/* 범례 */}
      <g>
        <line x1={HPAD + 5} y1={VPAD + 8} x2={HPAD + 20} y2={VPAD + 8} stroke="#e5e7eb" strokeWidth="1.5" />
        <text x={HPAD + 24} y={VPAD + 12} fill="#e5e7eb" fontSize="9">실제</text>
        <line x1={HPAD + 55} y1={VPAD + 8} x2={HPAD + 70} y2={VPAD + 8} stroke="#3b82f6" strokeWidth="1.5" strokeDasharray="5 3" />
        <text x={HPAD + 74} y={VPAD + 12} fill="#3b82f6" fontSize="9">HW 예측</text>
        {xgbForecast && (
          <>
            <line x1={HPAD + 125} y1={VPAD + 8} x2={HPAD + 140} y2={VPAD + 8} stroke="#f97316" strokeWidth="1.5" strokeDasharray="4 2" />
            <text x={HPAD + 144} y={VPAD + 12} fill="#f97316" fontSize="9">XGBoost</text>
          </>
        )}
      </g>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 분석 패널 메인 컴포넌트
// ─────────────────────────────────────────────────────────────────────────────
export function AnalysisPanel({ data, loading, error, market }: Props) {
  const [tab, setTab] = useState<'chart' | 'ai' | 'forecast' | 'news'>('chart');
  const isKrx = data?.market === 'KRX' || market === 'KRX';

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
        <p className="text-gray-400 text-sm">주가 데이터 분석 중...</p>
        <p className="text-gray-600 text-xs">AI 모델이 기술적 지표를 계산하고 있습니다</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="card max-w-md text-center">
          <div className="text-4xl mb-3">⚠️</div>
          <h3 className="text-lg font-semibold text-red-400 mb-2">분석 오류</h3>
          <p className="text-gray-400 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-6">
        <div className="text-6xl">📊</div>
        <h2 className="text-2xl font-bold text-white">주식 AI 예측 시스템</h2>
        <p className="text-gray-400 max-w-md">
          왼쪽 패널에서 종목명 또는 코드를 입력하고<br />
          <strong className="text-blue-400">분석 시작</strong> 버튼을 누르세요.
        </p>
        <div className="grid grid-cols-2 gap-3 mt-4 max-w-sm w-full">
          {['삼성전자', 'SK하이닉스', 'NVDA', 'TSLA'].map(s => (
            <div key={s} className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-2.5 text-sm text-gray-400 text-center">
              {s}
            </div>
          ))}
        </div>
      </div>
    );
  }

  const { symbol, company, last_close, pct_change, rsi, volume, atr, score,
    analysis_steps, candlestick_patterns, chart_data, forecast, xgb_forecast,
    risk_scenarios, news, naver } = data;

  const changePos = pct_change >= 0;
  const currency = isKrx ? '원' : '$';

  return (
    <div className="p-6 space-y-5">
      {/* ── 헤더 ── */}
      <div>
        <h2 className="text-2xl font-bold text-white">
          {company || symbol}
          <span className="ml-2 text-sm font-normal text-gray-400 bg-gray-800 px-2 py-0.5 rounded">
            {symbol}
          </span>
        </h2>
        <p className="text-xs text-gray-500 mt-1">
          기준일: {new Date().toLocaleDateString('ko-KR')} | 시장: {isKrx ? '🇰🇷 KRX (한국)' : '🇺🇸 US (미국)'}
        </p>
      </div>

      {/* ── 핵심 메트릭 ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="metric-card">
          <span className="metric-label">현재가</span>
          <span className="metric-value text-white">{fmt(last_close, isKrx)}</span>
          <span className={`metric-change ${changePos ? 'text-red-400' : 'text-blue-400'}`}>
            {changePos ? '▲' : '▼'} {Math.abs(pct_change).toFixed(2)}%
          </span>
        </div>
        <div className="metric-card">
          <span className="metric-label">RSI (14)</span>
          <span className={`metric-value ${rsi > 70 ? 'text-red-400' : rsi < 30 ? 'text-blue-400' : 'text-white'}`}>
            {rsi.toFixed(1)}
          </span>
          <span className="text-xs text-gray-500">
            {rsi > 70 ? '과매수' : rsi < 30 ? '과매도' : '중립'}
          </span>
        </div>
        <div className="metric-card">
          <span className="metric-label">거래량</span>
          <span className="metric-value text-white">{volume.toLocaleString()}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">ATR (변동성)</span>
          <span className="metric-value text-white">{atr.toLocaleString()}</span>
        </div>
      </div>

      {/* ── KRX 펀더멘털 ── */}
      {isKrx && naver && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">🏢 기업 펀더멘털 (네이버 금융)</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: '시가총액', value: naver.market_cap },
              { label: 'PER', value: naver.per },
              { label: 'PBR', value: naver.pbr },
              { label: '투자의견', value: naver.opinion },
            ].map(({ label, value }) => (
              <div key={label} className="bg-gray-800 rounded-xl p-3">
                <p className="text-xs text-gray-500">{label}</p>
                <p className="text-sm font-semibold text-white mt-1">{value || '-'}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── 탭 네비게이션 ── */}
      <div className="flex gap-2 border-b border-gray-800 pb-2">
        {([
          ['chart', '📊 차트 분석'],
          ['ai', '🧠 AI 진단'],
          ['forecast', '🔮 미래 예측'],
          ['news', '📰 뉴스/공시'],
        ] as const).map(([key, label]) => (
          <button
            key={key}
            className={`tab-btn ${tab === key ? 'active' : ''}`}
            onClick={() => setTab(key as any)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── 탭 콘텐츠 ── */}

      {/* 차트 탭 */}
      {tab === 'chart' && (
        <div className="card">
          <CandleChart
            chartData={chart_data}
            market={data.market}
            forecast={forecast}
            xgbForecast={xgb_forecast}
          />
        </div>
      )}

      {/* AI 진단 탭 */}
      {tab === 'ai' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="space-y-4">
            {/* 종합 점수 */}
            <div className="card">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">🏆 종합 기술적 점수</h3>
              <div className="flex items-end gap-3 mb-3">
                <span className={`text-5xl font-bold ${scoreColor(score)}`}>{score}</span>
                <span className="text-gray-500 text-lg mb-1">/ 100점</span>
              </div>
              <div className="score-bar-bg">
                <div
                  className={`score-bar-fill ${scoreBarColor(score)}`}
                  style={{ width: `${score}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-2">
                {score >= 70 ? '✅ 상승 우위 - 매수 신호 강함'
                 : score >= 40 ? '⚖️ 중립 - 추세 확인 필요'
                 : '⚠️ 하락 우위 - 하락 신호 주의'}
              </p>
            </div>

            {/* 캔들 패턴 */}
            <div className="card">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">🕯️ 캔들스틱 패턴</h3>
              {candlestick_patterns.length > 0 ? (
                <div className="space-y-2">
                  {candlestick_patterns.map((p, i) => (
                    <div
                      key={i}
                      className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm ${
                        p.direction === '상승' ? 'bg-red-950 border border-red-900' :
                        p.direction === '하락' ? 'bg-blue-950 border border-blue-900' :
                        'bg-gray-800 border border-gray-700'
                      }`}
                    >
                      <span className="font-medium">
                        {p.direction === '상승' ? '📈' : p.direction === '하락' ? '📉' : '➖'} {p.name}
                      </span>
                      <span className="text-xs text-gray-400">{p.desc}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500">특이한 캔들 패턴이 감지되지 않았습니다.</p>
              )}
            </div>
          </div>

          {/* 단계별 분석 */}
          <div className="card">
            <h3 className="text-sm font-semibold text-gray-300 mb-3">📝 단계별 분석 리포트</h3>
            <div className="space-y-3">
              {analysis_steps.map((step, i) => (
                <div key={i} className="bg-gray-800 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-white">{step.step}</span>
                    <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                      step.score > 0 ? 'bg-green-900 text-green-300' :
                      step.score < 0 ? 'bg-red-900 text-red-300' :
                      'bg-gray-700 text-gray-400'
                    }`}>
                      {step.score > 0 ? '+' : ''}{step.score}점
                    </span>
                  </div>
                  <p className="text-sm text-gray-400 leading-relaxed">{step.result}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 예측 탭 */}
      {tab === 'forecast' && (
        <div className="space-y-5">
          {/* 예측 차트 */}
          <div className="card">
            <h3 className="text-sm font-semibold text-gray-300 mb-3">🔮 가격 예측 (30일)</h3>
            <ForecastChart
              chartData={chart_data}
              forecast={forecast}
              xgbForecast={xgb_forecast}
              isKrx={isKrx}
            />

            {/* 예측 요약 */}
            {forecast && (
              <div className="grid grid-cols-3 gap-3 mt-4">
                {(() => {
                  const hwFinal = forecast.yhat[forecast.yhat.length - 1];
                  const xgbFinal = xgb_forecast ? xgb_forecast[xgb_forecast.length - 1] : null;
                  const ensemble = xgbFinal != null
                    ? hwFinal * 0.6 + xgbFinal * 0.4
                    : hwFinal;
                  const ensChg = (ensemble - last_close) / last_close * 100;

                  // 최적 매도 타이밍
                  let peakDate = '', peakPrice = 0;
                  if (xgb_forecast) {
                    const ensemblePreds = forecast.yhat.map((h, i) =>
                      h * 0.6 + (xgb_forecast[i] || h) * 0.4
                    );
                    const peakIdx = ensemblePreds.indexOf(Math.max(...ensemblePreds));
                    peakDate = forecast.dates[peakIdx] || '';
                    peakPrice = ensemblePreds[peakIdx];
                  }

                  return (
                    <>
                      <div className="bg-gray-800 rounded-xl p-3 text-center">
                        <p className="text-xs text-gray-500 mb-1">HW 예측 (30일)</p>
                        <p className="text-sm font-bold text-blue-400">{fmt(hwFinal, isKrx)}</p>
                      </div>
                      {xgbFinal != null && (
                        <div className="bg-gray-800 rounded-xl p-3 text-center">
                          <p className="text-xs text-gray-500 mb-1">XGBoost (30일)</p>
                          <p className="text-sm font-bold text-orange-400">{fmt(xgbFinal, isKrx)}</p>
                        </div>
                      )}
                      <div className="bg-blue-950 border border-blue-900 rounded-xl p-3 text-center">
                        <p className="text-xs text-gray-400 mb-1">🤖 AI 앙상블</p>
                        <p className="text-base font-bold text-white">{fmt(ensemble, isKrx)}</p>
                        <p className={`text-xs ${ensChg >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {ensChg >= 0 ? '▲' : '▼'} {Math.abs(ensChg).toFixed(2)}%
                        </p>
                      </div>
                    </>
                  );
                })()}
              </div>
            )}
          </div>

          {/* 리스크 관리 */}
          <div>
            <h3 className="text-sm font-semibold text-gray-300 mb-3">🛡️ 리스크 관리 (ATR 기반)</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(risk_scenarios).map(([key, scenario]) => {
                const borderColor = key === 'conservative' ? 'border-green-800'
                  : key === 'balanced' ? 'border-yellow-800' : 'border-red-800';
                const bgColor = key === 'conservative' ? 'bg-green-950'
                  : key === 'balanced' ? 'bg-yellow-950' : 'bg-red-950';
                return (
                  <div key={key} className={`risk-card ${bgColor} ${borderColor} border`}>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xl">{scenario.icon}</span>
                      <span className="font-semibold text-sm text-white">{scenario.label}</span>
                    </div>
                    <p className="text-xs text-gray-400 mb-3">{scenario.desc}</p>
                    <div className="space-y-1.5">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">🎯 목표가</span>
                        <span className="font-bold text-red-400">{fmt(scenario.target, isKrx)}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">🛑 손절가</span>
                        <span className="font-bold text-blue-400">{fmt(scenario.stop, isKrx)}</span>
                      </div>
                      <div className="flex justify-between text-xs mt-2 pt-2 border-t border-gray-700">
                        <span className="text-gray-500">손익비</span>
                        <span className="text-gray-300 font-medium">{scenario.ratio}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* 뉴스 탭 */}
      {tab === 'news' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* 네이버 뉴스 (KRX) */}
          {isKrx && naver ? (
            <>
              <div className="card">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">📰 주요 뉴스 (네이버)</h3>
                {naver.news.length > 0 ? (
                  <div className="space-y-2">
                    {naver.news.map((n, i) => (
                      <div key={i} className="news-item">
                        <span className="text-blue-400 mt-0.5 shrink-0">📄</span>
                        <a href={n.link} target="_blank" rel="noreferrer"
                          className="text-sm text-gray-300 hover:text-white hover:underline leading-relaxed">
                          {n.title}
                        </a>
                      </div>
                    ))}
                  </div>
                ) : <p className="text-sm text-gray-500">뉴스 없음</p>}
              </div>
              <div className="card">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">📋 최근 공시 (네이버)</h3>
                {naver.disclosures?.length > 0 ? (
                  <div className="space-y-2">
                    {naver.disclosures.map((d: any, i: number) => (
                      <div key={i} className="news-item">
                        <span className="text-yellow-400 mt-0.5 shrink-0">📌</span>
                        <a href={d.link} target="_blank" rel="noreferrer"
                          className="text-sm text-gray-300 hover:text-white hover:underline leading-relaxed">
                          {d.title}
                        </a>
                      </div>
                    ))}
                  </div>
                ) : <p className="text-sm text-gray-500">공시 없음</p>}
              </div>
            </>
          ) : (
            <div className="card lg:col-span-2">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">📰 관련 뉴스 (Google RSS)</h3>
              {news.length > 0 ? (
                <div className="space-y-2">
                  {news.map((n, i) => (
                    <div key={i} className="news-item">
                      <span className="text-blue-400 mt-0.5 shrink-0">🌐</span>
                      <div className="flex-1 min-w-0">
                        <a href={n.link} target="_blank" rel="noreferrer"
                          className="text-sm text-gray-300 hover:text-white hover:underline leading-relaxed block">
                          {n.title}
                        </a>
                        <p className="text-xs text-gray-600 mt-0.5">{n.publisher} · {n.published?.slice(0, 16)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500">검색된 뉴스가 없습니다.</p>
              )}
            </div>
          )}
        </div>
      )}

      {/* 면책 조항 */}
      <div className="border-t border-gray-800 pt-4">
        <p className="text-xs text-gray-600">
          ⚠️ 본 분석 리포트는 AI 모델 및 기술적 지표에 기반한 <strong>참고용 자료</strong>이며, 
          투자에 대한 최종 책임은 본인에게 있습니다.
        </p>
      </div>
    </div>
  );
}
