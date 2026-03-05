'use client';

import { useState, useEffect } from 'react';

interface ScreenerItem {
  market: string;
  name: string;
  ticker: string;
  price: string;
  change: number;
  category: string;
  volume: number;
}

interface ScreenerData {
  data: ScreenerItem[];
  usd_krw: number;
}

export function ScreenerPanel() {
  const [data, setData] = useState<ScreenerData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeMarket, setActiveMarket] = useState<'국내' | '해외'>('국내');
  const [sortKey, setSortKey] = useState<'change' | 'volume'>('change');
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc');

  const loadData = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/screener');
      const json = await res.json();
      if (json.error) {
        setError(json.error);
      } else {
        setData(json);
      }
    } catch {
      setError('스크리너 데이터를 불러올 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadData(); }, []);

  const filtered = data?.data
    .filter(s => s.market === activeMarket)
    .sort((a, b) => {
      const va = sortKey === 'change' ? a.change : a.volume;
      const vb = sortKey === 'change' ? b.change : b.volume;
      return sortDir === 'desc' ? vb - va : va - vb;
    }) || [];

  const handleSort = (key: typeof sortKey) => {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  return (
    <div className="p-6 space-y-5">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">📋 주식 골라보기</h2>
          <p className="text-xs text-gray-500 mt-1">
            국내/해외 주요 종목 실시간 시세
            {data && <span className="ml-2 text-gray-600">USD/KRW: {data.usd_krw.toLocaleString()}</span>}
          </p>
        </div>
        <button
          onClick={loadData}
          disabled={loading}
          className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-sm text-gray-300 rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? '⏳ 로딩 중...' : '🔄 새로고침'}
        </button>
      </div>

      {/* 마켓 탭 */}
      <div className="flex gap-2">
        {(['국내', '해외'] as const).map(m => (
          <button
            key={m}
            className={`tab-btn ${activeMarket === m ? 'active' : ''}`}
            onClick={() => setActiveMarket(m)}
          >
            {m === '국내' ? '🇰🇷 국내 (KRX)' : '🇺🇸 해외 (US)'}
          </button>
        ))}
      </div>

      {/* 에러 */}
      {error && (
        <div className="card border-red-900 bg-red-950">
          <p className="text-sm text-red-400">⚠️ {error}</p>
        </div>
      )}

      {/* 로딩 스켈레톤 */}
      {loading && !data && (
        <div className="space-y-2">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="h-12 bg-gray-800 rounded-xl animate-pulse" />
          ))}
        </div>
      )}

      {/* 테이블 */}
      {!loading && filtered.length > 0 && (
        <div className="card overflow-x-auto p-0">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wide w-8">#</th>
                <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wide">종목</th>
                <th className="px-4 py-3 text-right text-xs text-gray-500 uppercase tracking-wide">현재가</th>
                <th
                  className="px-4 py-3 text-right text-xs text-gray-500 uppercase tracking-wide cursor-pointer hover:text-white select-none"
                  onClick={() => handleSort('change')}
                >
                  등락률 {sortKey === 'change' ? (sortDir === 'desc' ? '↓' : '↑') : ''}
                </th>
                <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wide">카테고리</th>
                <th
                  className="px-4 py-3 text-right text-xs text-gray-500 uppercase tracking-wide cursor-pointer hover:text-white select-none"
                  onClick={() => handleSort('volume')}
                >
                  거래량 {sortKey === 'volume' ? (sortDir === 'desc' ? '↓' : '↑') : ''}
                </th>
                <th className="px-4 py-3 text-center text-xs text-gray-500 uppercase tracking-wide">신호</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s, i) => {
                const isUp = s.change >= 0;
                const isKrx = activeMarket === '국내';
                const signal = s.change > 3 ? '강력 매수' : s.change > 0 ? '매수' : s.change > -3 ? '중립' : '매도';
                const signalColor = signal === '강력 매수' ? 'text-green-400 bg-green-900'
                  : signal === '매수' ? 'text-green-300 bg-green-900'
                  : signal === '중립' ? 'text-yellow-300 bg-yellow-900'
                  : 'text-red-400 bg-red-900';

                return (
                  <tr
                    key={i}
                    className="border-b border-gray-800 hover:bg-gray-800 transition-colors"
                  >
                    <td className="px-4 py-3 text-gray-500 text-xs">{i + 1}</td>
                    <td className="px-4 py-3">
                      <div className="font-medium text-white">{s.name}</div>
                      <div className="text-xs text-gray-500">{s.ticker}</div>
                    </td>
                    <td className="px-4 py-3 text-right font-medium text-white">{s.price}</td>
                    <td className={`px-4 py-3 text-right font-bold ${
                      isKrx
                        ? (isUp ? 'text-red-400' : 'text-blue-400')
                        : (isUp ? 'text-green-400' : 'text-red-400')
                    }`}>
                      {isUp ? '▲' : '▼'} {Math.abs(s.change).toFixed(2)}%
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-xs px-2 py-0.5 rounded-full bg-gray-700 text-gray-300">
                        {s.category}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right text-gray-400 text-xs">
                      {s.volume.toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${signalColor}`}>
                        {signal}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {!loading && filtered.length === 0 && !error && (
        <div className="card text-center py-10">
          <p className="text-gray-500">데이터가 없습니다.</p>
        </div>
      )}

      <p className="text-xs text-gray-700">
        ⚠️ 등락률은 직전 거래일 종가 기준이며, 실시간 데이터가 아닐 수 있습니다.
      </p>
    </div>
  );
}
