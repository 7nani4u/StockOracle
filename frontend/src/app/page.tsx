'use client';

import { useState, useEffect, useRef } from 'react';
import { AnalysisPanel } from '../components/AnalysisPanel';
import { ScreenerPanel } from '../components/ScreenerPanel';

export default function Home() {
  const [page, setPage] = useState<'analysis' | 'screener'>('analysis');
  const [sentiment, setSentiment] = useState<any>(null);
  const [market, setMarket] = useState<'KRX' | 'US'>('KRX');
  const [ticker, setTicker] = useState('삼성전자');
  const [period, setPeriod] = useState('1y');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState('');

  // 시장 심리 로딩
  useEffect(() => {
    fetch(`/api/sentiment?market=${market}`)
      .then(r => r.json())
      .then(setSentiment)
      .catch(() => setSentiment(null));
  }, [market]);

  const handleAnalyze = async () => {
    if (!ticker.trim()) return;
    setLoading(true);
    setError('');
    setData(null);
    try {
      const res = await fetch(
        `/api/stock?ticker=${encodeURIComponent(ticker)}&period=${period}&market=${market}`
      );
      const json = await res.json();
      if (json.error) {
        setError(json.error);
      } else {
        setData(json);
        // 시장 자동 업데이트
        if (json.market) setMarket(json.market as 'KRX' | 'US');
      }
    } catch (e) {
      setError('API 서버에 연결할 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleAnalyze();
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* ── 사이드바 ── */}
      <aside className="w-72 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
        {/* 헤더 */}
        <div className="p-5 border-b border-gray-800">
          <h1 className="text-lg font-bold text-white flex items-center gap-2">
            📈 주식 AI 예측 시스템
          </h1>
          <p className="text-xs text-gray-500 mt-1">KRX / US 기술적 분석</p>
        </div>

        {/* 네비게이션 */}
        <nav className="p-4 border-b border-gray-800 space-y-1">
          <button
            className={`w-full text-left px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
              page === 'analysis'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white'
            }`}
            onClick={() => setPage('analysis')}
          >
            🔍 종목 상세 분석
          </button>
          <button
            className={`w-full text-left px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
              page === 'screener'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white'
            }`}
            onClick={() => setPage('screener')}
          >
            📋 주식 골라보기 (Screener)
          </button>
        </nav>

        {/* 시장 심리 위젯 */}
        <div className="p-4 border-b border-gray-800">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-3">🌍 시장 심리</p>
          {sentiment ? (
            <div className="bg-gray-800 rounded-xl p-3">
              <p className="text-xs text-gray-400">{sentiment.name}</p>
              <p className="text-xl font-bold text-white mt-1">
                {typeof sentiment.value === 'number' ? sentiment.value.toFixed(2) : '-'}
              </p>
              <p className={`text-sm font-medium mt-0.5 ${
                (sentiment.change ?? 0) >= 0 ? 'text-red-400' : 'text-blue-400'
              }`}>
                {(sentiment.change ?? 0) >= 0 ? '▲' : '▼'} {Math.abs(sentiment.change ?? 0).toFixed(2)}%
              </p>
              <span className="inline-block mt-2 text-xs px-2 py-0.5 rounded-full bg-gray-700 text-gray-300">
                {sentiment.sentiment}
              </span>
            </div>
          ) : (
            <div className="text-xs text-gray-600">로딩 중...</div>
          )}
        </div>

        {/* 분석 설정 (분석 탭에서만 표시) */}
        {page === 'analysis' && (
          <div className="p-4 flex flex-col gap-3 flex-1">
            <div>
              <label className="text-xs text-gray-500 uppercase tracking-wide mb-1.5 block">
                시장 선택
              </label>
              <div className="flex gap-2">
                {(['KRX', 'US'] as const).map(m => (
                  <button
                    key={m}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${
                      market === m
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                    onClick={() => setMarket(m)}
                  >
                    {m === 'KRX' ? '🇰🇷 한국' : '🇺🇸 미국'}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-xs text-gray-500 uppercase tracking-wide mb-1.5 block">
                종목명 / 코드
              </label>
              <input
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
                placeholder={market === 'KRX' ? '예: 삼성전자, 005930' : '예: 애플, TSLA'}
                value={ticker}
                onChange={e => setTicker(e.target.value)}
                onKeyDown={handleKeyDown}
              />
            </div>

            <div>
              <label className="text-xs text-gray-500 uppercase tracking-wide mb-1.5 block">
                분석 기간
              </label>
              <select
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500"
                value={period}
                onChange={e => setPeriod(e.target.value)}
              >
                <option value="6mo">6개월</option>
                <option value="1y">1년</option>
                <option value="2y">2년</option>
                <option value="5y">5년</option>
              </select>
            </div>

            <button
              className="w-full bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white font-semibold py-3 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed mt-auto"
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  분석 중...
                </span>
              ) : (
                '🔍 분석 시작'
              )}
            </button>
          </div>
        )}

        {/* 하단 면책 */}
        <div className="p-4 mt-auto border-t border-gray-800">
          <p className="text-[10px] text-gray-600 leading-relaxed">
            ⚠️ 본 시스템은 참고용이며, 투자 결정의 책임은 본인에게 있습니다.
          </p>
        </div>
      </aside>

      {/* ── 메인 콘텐츠 ── */}
      <main className="flex-1 overflow-y-auto bg-gray-950">
        {page === 'screener' ? (
          <ScreenerPanel />
        ) : (
          <AnalysisPanel
            data={data}
            loading={loading}
            error={error}
            market={market}
          />
        )}
      </main>
    </div>
  );
}
