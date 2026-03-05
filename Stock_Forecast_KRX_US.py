# -*- coding: utf-8 -*-
"""
Stock AI Prediction System (KRX/US) - v4.1 (Final)
Advanced Version with Prophet, XGBoost Ensemble, Sentiment Analysis (RSS), and Sell Timing
"""

import streamlit as st
import pandas as pd
import numpy as np
# Monkeypatch platformdirs to use local cache directory to avoid sqlite locking issues
try:
    import platformdirs
    import os
    def get_custom_cache_dir(appname=None, *args, **kwargs):
        # Use a local directory for cache
        cache_dir = os.path.join(os.getcwd(), ".yf_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    platformdirs.user_cache_dir = get_custom_cache_dir
except ImportError:
    pass

import yfinance as yf
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import time
import requests
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# Helper Functions for Ticker Resolution (Name -> Code)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600*24) # Cache for 24 hours
def get_krx_code_map():
    """
    Fetch KRX stock list from KIND and return a Name -> Code mapping.
    Also returns a Code -> Name mapping.
    """
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
    try:
        res = requests.get(url)
        res.encoding = 'euc-kr' 
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.select_one("table")
        
        name_to_code = {}
        code_to_name = {}
        
        if table:
            rows = table.select("tr")
            for row in rows[1:]: # Skip header
                cols = row.select("td")
                if len(cols) >= 3:
                    name = cols[0].text.strip()
                    code = cols[2].text.strip()
                    
                    # Pad code to 6 digits
                    code = code.zfill(6)
                    
                    name_to_code[name] = code
                    code_to_name[code] = name
                    
        return name_to_code, code_to_name
    except Exception as e:
        # Fallback if KIND is down
        return {}, {}

# Hardcoded Mappings for User Convenience
COMMON_ALIASES = {
    # KRX
    "삼성": "005930",
    "삼전": "005930",
    "하이닉스": "000660",
    "카카오": "035720",
    "네이버": "035420",
    "현대차": "005380",
    "기아": "000270",
    "엘지": "003550",
    "LG": "003550",
    "포스코": "005490",
    "셀트리온": "068270",
    "KB금융": "105560",
    "신한지주": "055550",
    "SK": "034730",
    "SK하이닉스": "000660",
    "LG에너지솔루션": "373220",
    "엔솔": "373220",
    "두산에너빌리티": "034020",
    "에코프로": "086520",
    "에코프로비엠": "247540",
}

US_STOCK_MAPPING = {
    # Popular US Stocks (Korean Name -> Ticker)
    "애플": "AAPL",
    "테슬라": "TSLA",
    "마이크로소프트": "MSFT",
    "마소": "MSFT",
    "엔비디아": "NVDA",
    "아마존": "AMZN",
    "구글": "GOOGL",
    "알파벳": "GOOGL",
    "메타": "META",
    "페이스북": "META",
    "넷플릭스": "NFLX",
    "AMD": "AMD",
    "인텔": "INTC",
    "코카콜라": "KO",
    "펩시": "PEP",
    "스타벅스": "SBUX",
    "나이키": "NKE",
    "디즈니": "DIS",
    "맥도날드": "MCD",
    "코스트코": "COST",
    "월마트": "WMT",
    "제이피모건": "JPM",
    "비자": "V",
    "마스터카드": "MA",
    "화이자": "PFE",
    "모더나": "MRNA",
    "TSMC": "TSM",
    "알리바바": "BABA",
    "쿠팡": "CPNG",
    "로블록스": "RBLX",
    "유니티": "U",
    "팔란티어": "PLTR",
    "코인베이스": "COIN",
    "게임스탑": "GME",
    "AMC": "AMC",
    "QQQ": "QQQ",
    "SPY": "SPY",
    "TQQQ": "TQQQ",
    "SOXL": "SOXL",
    "SQQQ": "SQQQ",
    "비트코인": "BTC-USD",
    "이더리움": "ETH-USD",
}

def resolve_ticker(user_input):
    """
    Resolve user input (Name or Code) to a valid Ticker.
    Returns: (ticker, market_type, company_name) or (None, None, None)
    """
    user_input = user_input.strip()
    if not user_input:
        return None, None, None
        
    # 1. Check if it's a known Alias
    if user_input in COMMON_ALIASES:
        code = COMMON_ALIASES[user_input]
        return f"{code}.KS", "KRX", user_input # Default to KS for now, check later if KQ
    
    # 2. Check US Mapping
    if user_input in US_STOCK_MAPPING:
        return US_STOCK_MAPPING[user_input], "US", user_input
        
    # 3. Check if it looks like a KRX Ticker (6 digits)
    if user_input.isdigit() and len(user_input) == 6:
        # Default to .KS, but we should verify. 
        # Ideally check against full KRX list to see if it exists.
        name_map, code_map = get_krx_code_map()
        if user_input in code_map:
            return f"{user_input}.KS", "KRX", code_map[user_input]
        return f"{user_input}.KS", "KRX", user_input
        
    # 4. Check if it looks like a US Ticker (English chars)
    # Exclude if it's Korean
    if all(ord(c) < 128 for c in user_input):
        # Assume US Ticker if pure English/Numbers and not 6 digits
        return user_input.upper(), "US", user_input.upper()
        
    # 5. Search in Full KRX List
    name_map, code_map = get_krx_code_map()
    if user_input in name_map:
        code = name_map[user_input]
        return f"{code}.KS", "KRX", user_input
        
    # 6. Fuzzy Search / Partial Match in KRX List
    # e.g. User types "삼성전자" (exact match handled above)
    # User types "삼성전자우"
    for name, code in name_map.items():
        if user_input in name: # Simple contains check
            # Return the first match? Or maybe the one that starts with?
            # Prefer "Starts With"
            if name.startswith(user_input):
                 return f"{code}.KS", "KRX", name
                 
    # If still not found, return None
    return None, None, None
import warnings
import requests
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# 1. Library Imports & Configuration
# -----------------------------------------------------------------------------
# Try importing optional libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    
# Suppress warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="주식 AI 예측 시스템 (KRX/US)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling (Enhanced)
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .stMetric {
        background-color: #F8F9FA;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #E9ECEF;
    }
    .big-font { font-size: 24px !important; font-weight: bold; }
    
    /* Risk Cards */
    .risk-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .risk-card:hover {
        transform: translateY(-5px);
    }
    .risk-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .risk-desc {
        font-size: 13px;
        color: #666;
        margin-bottom: 15px;
        min-height: 40px;
    }
    .risk-price {
        font-size: 15px;
        margin: 5px 0;
        display: flex;
        justify-content: space-between;
    }
    .risk-label {
        font-weight: 600;
        color: #444;
    }
    
    /* Analysis Section */
    .analysis-box {
        background-color: #f8f9fa;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .analysis-step {
        margin-bottom: 10px;
    }
    .step-title {
        font-weight: bold;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data Fetching & Processing
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_naver_finance_data(code):
    """
    네이버 금융에서 실시간 주가, 기본 정보, 뉴스/공시, 기업실적 등을 크롤링
    """
    code = str(code).replace('.KS', '').replace('.KQ', '')
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    
    data = {
        'price': None,
        'market_cap': None,
        'per': None,
        'eps': None,
        'pbr': None,
        'dividend_yield': None,
        'opinion': None,
        'target_price': None,
        'news': [],
        'disclosures': [],
        'financials': {}
    }
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 1. Price
        try:
            # blind element inside no_today
            price_elem = soup.select_one(".no_today .blind")
            if price_elem:
                data['price'] = price_elem.text.replace(',', '')
        except:
            pass
            
        # 2. Fundamentals
        try:
            data['market_cap'] = soup.select_one("#_market_sum").text.strip().replace('\t', '').replace('\n', '') if soup.select_one("#_market_sum") else "-"
            data['per'] = soup.select_one("#_per").text if soup.select_one("#_per") else "-"
            data['eps'] = soup.select_one("#_eps").text if soup.select_one("#_eps") else "-"
            data['pbr'] = soup.select_one("#_pbr").text if soup.select_one("#_pbr") else "-"
            data['dividend_yield'] = soup.select_one("#_dvr").text if soup.select_one("#_dvr") else "-"
        except:
            pass
            
        # 3. Investment Opinion
        try:
            # Usually in .rgt > .f_total or similar.
            # Consenus info is tricky, let's try generic selector
            opinion_elem = soup.select_one(".rgt .f_total .f_up") or soup.select_one(".rgt .f_total .f_down") or soup.select_one(".rgt .f_total .f_eq")
            if opinion_elem:
                 data['opinion'] = opinion_elem.text.strip()
            
            tp_elem = soup.select_one(".rgt .f_total em")
            if tp_elem:
                data['target_price'] = tp_elem.text.strip()
        except:
            pass
            
        # 4. News (Extract from main page news section)
        try:
            news_section = soup.select(".news_section ul li")
            for item in news_section[:5]: # Top 5
                a_tag = item.select_one("span > a")
                if a_tag:
                    title = a_tag.text.strip()
                    link = "https://finance.naver.com" + a_tag['href']
                    # Date provider often in separate span or implicit
                    data['news'].append({'title': title, 'link': link})
        except:
            pass
            
        # 5. Disclosures (Notice section)
        try:
            notice_section = soup.select(".notice_section ul li")
            for item in notice_section[:5]:
                a_tag = item.select_one("span > a")
                if a_tag:
                    title = a_tag.text.strip()
                    link = "https://finance.naver.com" + a_tag['href']
                    data['disclosures'].append({'title': title, 'link': link})
        except:
            pass
            
    except Exception as e:
        pass
        
    return data

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, market, period="2y", interval="1d"):
    """
    주식 데이터 가져오기 (yfinance) 및 뉴스 (Google RSS)
    """
    symbol = ticker.strip().upper()
    
    # KRX 심볼 처리
    if market == "KRX (한국)":
        if symbol.isdigit():
            symbol = f"{symbol}.KS"  # 기본 코스피
            
    try:
        # Ticker 객체 생성
        ticker_obj = yf.Ticker(symbol)
        data = ticker_obj.history(period=period, interval=interval)
        
        # 코스피 데이터 없으면 코스닥 시도
        if data.empty:
            if market == "KRX (한국)" and symbol.endswith(".KS"):
                symbol = symbol.replace(".KS", ".KQ")
                ticker_obj = yf.Ticker(symbol)
                data = ticker_obj.history(period=period, interval=interval)
        
        if data.empty:
            return None, None, f"데이터를 찾을 수 없습니다: {symbol}"
            
        # 컬럼 정리
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        # 기술적 지표 계산
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA60'] = data['Close'].rolling(window=60).mean()
        data['MA120'] = data['Close'].rolling(window=120).mean()
        
        # EMA
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_Std'])
        data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_Std'])
        
        # ATR (Average True Range) for Risk Management
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        
        # Stochastic Oscillator
        low_min = data['Low'].rolling(window=14).min()
        high_max = data['High'].rolling(window=14).max()
        data['%K'] = (data['Close'] - low_min) / (high_max - low_min) * 100
        data['%D'] = data['%K'].rolling(window=3).mean()

        # News fetching (Google News RSS via feedparser)
        news = []
        if FEEDPARSER_AVAILABLE:
            try:
                # KRX vs US RSS URL
                if market == "KRX (한국)":
                    # Use Google News KR for specific query
                    # Remove .KS/.KQ for search query
                    query_symbol = symbol.replace('.KS', '').replace('.KQ', '')
                    query = f"{query_symbol} 주가"
                    rss_url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
                else:
                    # US News
                    query = f"{symbol} stock"
                    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
                
                feed = feedparser.parse(rss_url)
                for entry in feed.entries[:5]:
                    news.append({
                        'title': entry.title,
                        'link': entry.link,
                        'publisher': entry.source.title if hasattr(entry, 'source') else 'Google News',
                        'published': entry.published if hasattr(entry, 'published') else datetime.datetime.now().strftime("%Y-%m-%d")
                    })
            except Exception as e:
                pass # Fail silently for news
        
        return data, news, symbol
        
    except Exception as e:
        return None, None, str(e)

@st.cache_data(ttl=3600)
def fetch_market_sentiment(market):
    """
    시장 전체 심리 (VIX or KS200)
    """
    try:
        if market == "US (미국)":
            ticker = "^VIX"
            name = "VIX (공포지수)"
        else:
            ticker = "^KS200" # KOSPI 200
            name = "KOSPI 200"
            
        data = yf.Ticker(ticker).history(period="1mo")
        if data.empty:
            return None
            
        current = data['Close'].iloc[-1]
        prev = data['Close'].iloc[-2]
        change = (current - prev) / prev * 100
        
        sentiment = "중립"
        if market == "US (미국)":
            if current > 30: sentiment = "극도의 공포"
            elif current > 20: sentiment = "공포/불안"
            elif current < 15: sentiment = "탐욕/안정"
        else:
            # KOSPI Logic (Simple Trend)
            if change > 1.0: sentiment = "강세장"
            elif change < -1.0: sentiment = "약세장"
            
        return {
            'name': name,
            'value': current,
            'change': change,
            'sentiment': sentiment
        }
    except:
        return None

# -----------------------------------------------------------------------------
# 3. AI Analysis Logic (Chain-of-Thought Style)
# -----------------------------------------------------------------------------
def analyze_trend_score(df):
    """
    기술적 지표 기반 종합 점수 계산 (0~100점) 및 상세 분석
    """
    score = 50
    analysis_steps = []
    
    current = df.iloc[-1]
    
    # Step 1: Trend Analysis (이동평균)
    trend_score = 0
    trend_msg = ""
    if current['Close'] > current['MA20']:
        trend_score += 10
        if current['Close'] > current['MA60']:
            trend_score += 10
            if current['MA20'] > current['MA60']: # 정배열
                trend_score += 10
                trend_msg = "단기/장기 이동평균선이 정배열 상태로, 강력한 상승 추세를 보이고 있습니다."
            else:
                trend_msg = "주가가 장기 이평선 위에 위치하여 상승 기조를 유지 중입니다."
        else:
            trend_msg = "단기적으로 20일 이평선을 상회하며 상승 시도 중입니다."
    else:
        trend_score -= 10
        if current['Close'] < current['MA60']:
            trend_score -= 10
            if current['MA20'] < current['MA60']: # 역배열
                trend_score -= 10
                trend_msg = "단기/장기 이동평균선이 역배열 상태로, 하락 압력이 강합니다."
            else:
                trend_msg = "주가가 장기 이평선 아래로 내려가 하락 추세가 우려됩니다."
        else:
            trend_msg = "단기적으로 20일 이평선을 하회하며 조정을 받고 있습니다."
            
    score += trend_score
    analysis_steps.append({"step": "1. 추세 분석 (Trend)", "result": trend_msg, "score": trend_score})
    
    # Step 2: Momentum Analysis (RSI, MACD)
    momentum_score = 0
    momentum_msg = []
    
    # RSI
    if current['RSI'] > 70:
        momentum_score -= 5
        momentum_msg.append(f"RSI가 {current['RSI']:.1f}로 과매수 구간입니다. 차익 실현 매물이 나올 수 있습니다.")
    elif current['RSI'] < 30:
        momentum_score += 10
        momentum_msg.append(f"RSI가 {current['RSI']:.1f}로 과매도 구간입니다. 기술적 반등이 기대됩니다.")
    else:
        momentum_msg.append(f"RSI는 {current['RSI']:.1f}로 중립적입니다.")
        
    # MACD
    if current['MACD'] > current['Signal_Line']:
        momentum_score += 10
        if current['MACD'] > 0:
            momentum_msg.append("MACD가 시그널 선 위에 있으며, 0선 위에서 상승 추세를 강화하고 있습니다.")
        else:
            momentum_msg.append("MACD 골든크로스가 발생하여 상승 전환 신호를 보냅니다.")
    else:
        momentum_score -= 10
        momentum_msg.append("MACD가 시그널 선 아래에 위치하여 하락 모멘텀이 우세합니다.")
        
    score += momentum_score
    analysis_steps.append({"step": "2. 모멘텀 분석 (Momentum)", "result": " ".join(momentum_msg), "score": momentum_score})

    # Step 3: Volatility & Volume (Bollinger, Volume)
    vol_score = 0
    vol_msg = []
    
    # Bollinger
    if current['Close'] > current['BB_Upper'] * 0.98:
        vol_score += 5
        vol_msg.append("볼린저 밴드 상단을 터치하며 강한 변동성을 보입니다.")
    elif current['Close'] < current['BB_Lower'] * 1.02:
        vol_score -= 5
        vol_msg.append("볼린저 밴드 하단을 터치하며 약세를 보입니다.")
        
    # Volume
    vol_mean = df['Volume'].iloc[-20:].mean()
    if current['Volume'] > vol_mean * 1.5:
        if current['Close'] > current['Open']:
            vol_score += 10
            vol_msg.append("평소 대비 1.5배 이상의 거래량을 동반한 상승입니다 (신뢰도 높음).")
        else:
            vol_score -= 10
            vol_msg.append("평소 대비 1.5배 이상의 거래량을 동반한 하락입니다 (매도 압력 강함).")
    else:
        vol_msg.append("거래량은 평이한 수준입니다.")
        
    score += vol_score
    analysis_steps.append({"step": "3. 거래량 및 변동성 (Volume/Volatility)", "result": " ".join(vol_msg), "score": vol_score})
    
    # Step 4: Pattern Analysis (TA-Lib)
    pattern_score = 0
    pattern_msg = []
    
    # TA-Lib 패턴 감지
    patterns = detect_candlestick_patterns_talib(df)
    
    if patterns:
        bullish_count = sum(1 for p in patterns if p['direction'] == '상승')
        bearish_count = sum(1 for p in patterns if p['direction'] == '하락')
        
        if bullish_count > bearish_count:
            pattern_score += 10
            pattern_msg.append(f"상승 패턴 {bullish_count}개가 감지되었습니다 (예: {patterns[0]['desc']}).")
        elif bearish_count > bullish_count:
            pattern_score -= 10
            pattern_msg.append(f"하락 패턴 {bearish_count}개가 감지되었습니다 (예: {patterns[0]['desc']}).")
        else:
            pattern_msg.append(f"상승/하락 패턴이 혼재되어 있습니다 ({len(patterns)}개 감지).")
            
        # 신뢰도 가중치 추가
        for p in patterns[:3]: # 상위 3개만
            if p['conf'] > 80:
                if p['direction'] == '상승': pattern_score += 5
                elif p['direction'] == '하락': pattern_score -= 5
    else:
        pattern_msg.append("특이한 캔들 패턴이 감지되지 않았습니다.")
        
    score += pattern_score
    analysis_steps.append({"step": "4. 캔들 패턴 분석 (TA-Lib 61)", "result": " ".join(pattern_msg), "score": pattern_score})
    
    return max(0, min(100, score)), analysis_steps, patterns # Return patterns too for UI display

def calculate_risk_scenarios(current_price, atr):
    """
    ATR 기반 손절/익절 시나리오 계산
    """
    if np.isnan(atr) or atr == 0:
        atr = current_price * 0.02 # Fallback: 2%
        
    scenarios = {
        '보수적 (Conservative)': {
            'target': current_price + (atr * 1.5),
            'stop': current_price - (atr * 1.0),
            'ratio': '1 : 1.5',
            'desc': '짧은 호흡, 리스크 최소화 전략',
            'color': '#E8F5E9',
            'icon': '🛡️'
        },
        '중립적 (Balanced)': {
            'target': current_price + (atr * 2.5),
            'stop': current_price - (atr * 1.5),
            'ratio': '1 : 1.67',
            'desc': '균형 잡힌 스윙 트레이딩 전략',
            'color': '#FFF3E0',
            'icon': '⚖️'
        },
        '공격적 (Aggressive)': {
            'target': current_price + (atr * 4.0),
            'stop': current_price - (atr * 2.0),
            'ratio': '1 : 2.0',
            'desc': '추세 추종, 큰 수익 목표 전략',
            'color': '#FFEBEE',
            'icon': '🚀'
        }
    }
    return scenarios

# -----------------------------------------------------------------------------
# 4. Pattern Recognition (TA-Lib)
# -----------------------------------------------------------------------------
def detect_candlestick_patterns_talib(df: pd.DataFrame) -> list:
    """TA-Lib 기반 61개 패턴 감지"""
    if not TALIB_AVAILABLE:
        return []
        
    patterns = []
    
    if len(df) < 5:  # 최소 5개 필요
        return []
    
    # 데이터 준비 (TA-Lib은 numpy array 필요)
    open_prices = df['Open'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values
    
    # TA-Lib 패턴 정의 (61개)
    pattern_functions = {
        # 단일(1-캔들) - 15개
        'CDLBELTHOLD': ('🔨 Belt Hold', '벨트 홀드', '단일'),
        'CDLCLOSINGMARUBOZU': ('📊 Closing Marubozu', '종가 마루보즈', '단일'),
        'CDLMARUBOZU': ('📏 Marubozu', '마루보즈', '단일'),
        'CDLLONGLINE': ('📐 Long Line', '장대봉', '단일'),
        'CDLSHORTLINE': ('📌 Short Line', '단봉', '단일'),
        'CDLSPINNINGTOP': ('🌪️ Spinning Top', '팽이형', '단일'),
        'CDLHIGHWAVE': ('🌊 High Wave', '높은 파동형', '단일'),
        'CDLHAMMER': ('🔨 Hammer', '해머', '단일'),
        'CDLHANGINGMAN': ('👤 Hanging Man', '교수형', '단일'),
        'CDLINVERTEDHAMMER': ('🔧 Inverted Hammer', '역망치', '단일'),
        'CDLSHOOTINGSTAR': ('⭐ Shooting Star', '유성형', '단일'),
        'CDLRICKSHAWMAN': ('🚶 Rickshaw Man', '릭샤맨', '단일'),
        'CDLTAKURI': ('🎣 Takuri', '타쿠리', '단일'),
        'CDLKICKING': ('👟 Kicking', '킥킹', '단일'),
        'CDLKICKINGBYLENGTH': ('👢 Kicking by Length', '킥킹(길이 기준)', '단일'),
        
        # 2-캔들 - 12개
        'CDLENGULFING': ('🫂 Engulfing', '포용형', '2-캔들'),
        'CDLHARAMI': ('🤰 Harami', '하라미', '2-캔들'),
        'CDLHARAMICROSS': ('➕ Harami Cross', '하라미 크로스', '2-캔들'),
        'CDLPIERCING': ('🎯 Piercing', '관통형', '2-캔들'),
        'CDLDARKCLOUDCOVER': ('☁️ Dark Cloud Cover', '암운형', '2-캔들'),
        'CDLCOUNTERATTACK': ('⚔️ Counterattack', '반격선', '2-캔들'),
        'CDLONNECK': ('🦢 On Neck', '온넥', '2-캔들'),
        'CDLINNECK': ('🦆 In Neck', '인넥', '2-캔들'),
        'CDLTHRUSTING': ('🗡️ Thrusting', '스러스팅', '2-캔들'),
        'CDLSEPARATINGLINES': ('↔️ Separating Lines', '세퍼레이팅 라인', '2-캔들'),
        'CDLMATCHINGLOW': ('🎯 Matching Low', '매칭 로우', '2-캔들'),
        'CDLHOMINGPIGEON': ('🕊️ Homing Pigeon', '호밍 피전', '2-캔들'),
        
        # 3-캔들 - 11개
        'CDL2CROWS': ('🐦 Two Crows', '투 크로우즈', '3-캔들'),
        'CDL3INSIDE': ('📦 Three Inside', '삼내부', '3-캔들'),
        'CDL3OUTSIDE': ('📤 Three Outside', '삼외부', '3-캔들'),
        'CDL3LINESTRIKE': ('⚡ Three Line Strike', '쓰리 라인 스트라이크', '3-캔들'),
        'CDL3BLACKCROWS': ('🐦‍⬛ Three Black Crows', '세 검은 까마귀', '3-캔들'),
        'CDLIDENTICAL3CROWS': ('🦅 Identical Three Crows', '동일 삼까마귀', '3-캔들'),
        'CDLUNIQUE3RIVER': ('🏞️ Unique 3 River', '유니크 쓰리 리버', '3-캔들'),
        'CDL3STARSINSOUTH': ('⭐ Three Stars in South', '남쪽의 세 별', '3-캔들'),
        'CDLUPSIDEGAP2CROWS': ('📈 Upside Gap Two Crows', '업사이드 갭 투 크로우즈', '3-캔들'),
        'CDLEVENINGSTAR': ('🌆 Evening Star', '석별형', '3-캔들'),
        'CDLTRISTAR': ('✨ Tristar', '트리스타', '3-캔들'),
        
        # 갭/지속/복합 - 9개
        'CDLBREAKAWAY': ('🚀 Breakaway', '브레이크어웨이', '복합'),
        'CDLRISEFALL3METHODS': ('📊 Rising/Falling 3 Methods', '상승하락 삼법', '복합'),
        'CDLMATHOLD': ('🤝 Mat Hold', '매트 홀드', '복합'),
        'CDLTASUKIGAP': ('📏 Tasuki Gap', '타스키 갭', '복합'),
        'CDLGAPSIDESIDEWHITE': ('⬜ Gap Side-by-Side White', '갭 사이드바이사이드', '복합'),
        'CDLXSIDEGAP3METHODS': ('📈 Gap Three Methods', '갭 쓰리 메서즈', '복합'),
        'CDLABANDONEDBABY': ('👶 Abandoned Baby', '어밴던드 베이비', '복합'),
        'CDLCONCEALBABYSWALL': ('🐦 Concealing Baby Swallow', '컨실링 베이비', '복합'),
        'CDLLADDERBOTTOM': ('🪜 Ladder Bottom', '래더 바텀', '복합'),
        
        # 특수 - 5개
        'CDLADVANCEBLOCK': ('🚧 Advance Block', '전진 봉쇄', '특수'),
        'CDLSTALLEDPATTERN': ('⏸️ Stalled Pattern', '정체 패턴', '특수'),
        'CDLSTICKSANDWICH': ('🥪 Stick Sandwich', '스틱 샌드위치', '특수'),
        'CDLHIKKAKE': ('🎣 Hikkake', '힛카케', '특수'),
        'CDLHIKKAKEMOD': ('🎯 Modified Hikkake', '수정 힛카케', '특수'),
        
        # 기존 3개 (TA-Lib에도 있지만 명시적으로 추가)
        'CDL3WHITESOLDIERS': ('⚪ Three White Soldiers', '세 개의 연속 양봉', '3-캔들'),
        'CDLMORNINGSTAR': ('🌅 Morning Star', '하락 후 반전 신호', '3-캔들'),
        'CDLDOJI': ('✖️ Doji', '매수/매도 균형', '단일'),
    }
    
    # 각 패턴 감지
    for func_name, (emoji_name, korean_name, category) in pattern_functions.items():
        try:
            if not hasattr(talib, func_name):
                continue
                
            pattern_func = getattr(talib, func_name)
            result = pattern_func(open_prices, high_prices, low_prices, close_prices)
            
            # 패턴 발생 지점 찾기 (마지막 날짜 기준)
            last_value = result[-1]
            if last_value != 0:
                # 신뢰도 변환: -100~100 → 0~100%
                confidence = abs(last_value)
                
                # 방향 판단
                if last_value > 0:
                    direction = '상승'
                    impact = '상승 신호'
                elif last_value < 0:
                    direction = '하락'
                    impact = '하락 신호'
                else:
                    direction = '중립'
                    impact = '추세 전환 가능성'
                
                patterns.append({
                    'name': emoji_name,
                    'category': category,
                    'conf': confidence,
                    'desc': korean_name,
                    'impact': impact,
                    'direction': direction,
                    'signal_value': last_value # for scoring
                })
        except Exception:
            continue
            
    return patterns

# -----------------------------------------------------------------------------
# 5. Chart Pattern Class (Geometric)
# -----------------------------------------------------------------------------
class ChartPatternAnalyzer:
    def __init__(self, df):
        self.df = df
        self.closes = df['Close'].values
        self.highs = df['High'].values
        self.lows = df['Low'].values
        
    def find_local_extrema(self, order=5):
        peaks_idx = argrelextrema(self.highs, np.greater, order=order)[0]
        troughs_idx = argrelextrema(self.lows, np.less, order=order)[0]
        return peaks_idx, troughs_idx

    def detect_patterns(self):
        patterns = []
        peaks, troughs = self.find_local_extrema(order=5)
        
        if len(peaks) < 3 or len(troughs) < 3:
            return patterns

        last_peaks = peaks[-3:]
        last_troughs = troughs[-3:]
        
        try:
            z_upper = np.polyfit(last_peaks, self.highs[last_peaks], 1)
            slope_upper = z_upper[0]
            z_lower = np.polyfit(last_troughs, self.lows[last_troughs], 1)
            slope_lower = z_lower[0]
            
            # Triangle Patterns
            if slope_upper < 0 and slope_lower > 0:
                patterns.append({'name': '대칭 삼각형 (Symmetrical Triangle)', 'signal': '중립/변동성 축소', 'desc': '곧 큰 방향성이 나올 것입니다.'})
            elif slope_upper < 0 and abs(slope_lower) < 0.05:
                patterns.append({'name': '하락 삼각형 (Descending Triangle)', 'signal': '매도 (하락형)', 'desc': '지지선 붕괴 위험이 있습니다.'})
            elif abs(slope_upper) < 0.05 and slope_lower > 0:
                patterns.append({'name': '상승 삼각형 (Ascending Triangle)', 'signal': '매수 (상승형)', 'desc': '저항선 돌파 시도가 예상됩니다.'})
                
            # Wedge Patterns
            if slope_upper < 0 and slope_lower < 0:
                if slope_lower < slope_upper:
                    patterns.append({'name': '하락 쐐기형 (Falling Wedge)', 'signal': '매수 (반전)', 'desc': '하락세가 약화되고 반등할 가능성이 큽니다.'})
            if slope_upper > 0 and slope_lower > 0:
                if slope_lower > slope_upper:
                    patterns.append({'name': '상승 쐐기형 (Rising Wedge)', 'signal': '매도 (반전)', 'desc': '상승세가 약화되고 하락할 가능성이 큽니다.'})
                    
            # Double Top/Bottom
            if len(last_peaks) >= 2:
                if abs(self.highs[last_peaks[-1]] - self.highs[last_peaks[-2]]) / self.highs[last_peaks[-1]] < 0.02:
                    patterns.append({'name': '이중 천장 (Double Top)', 'signal': '매도', 'desc': '고점 돌파 실패, 하락 전환 가능성.'})
            if len(troughs) >= 2:
                if abs(self.lows[last_troughs[-1]] - self.lows[last_troughs[-2]]) / self.lows[last_troughs[-1]] < 0.02:
                    patterns.append({'name': '이중 바닥 (Double Bottom)', 'signal': '매수', 'desc': '바닥 지지 성공, 상승 전환 가능성.'})

        except Exception:
            pass
        return patterns

# -----------------------------------------------------------------------------
# 5. XGBoost Model
# -----------------------------------------------------------------------------
def run_xgboost_forecast(df, forecast_days=30):
    if not XGBOOST_AVAILABLE:
        return None
        
    try:
        # Prepare Data
        data = df.copy()
        data['Target'] = data['Close'].shift(-1) # Next day price
        
        # Features
        data['Returns'] = data['Close'].pct_change()
        data['Range'] = data['High'] - data['Low']
        for lag in [1, 2, 3, 5]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            
        data = data.dropna()
        
        X = data[['Close', 'Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'Returns', 'Range']]
        y = data['Target']
        
        # Train/Test Split (Simple)
        split = int(len(X) * 0.95)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05)
        model.fit(X_train, y_train)
        
        # Recursive Forecast
        # Start from the last available data point
        last_row = X.iloc[[-1]].copy()
        predictions = []
        
        # We need to maintain a running "Close" to update lags
        # Note: This is a simplified recursive strategy. 
        # Ideally we re-calculate technical indicators (MA, RSI) but that requires full history.
        # Here we only update price-based lags for simplicity.
        
        current_close = df['Close'].iloc[-1]
        
        for _ in range(forecast_days):
            pred_close = model.predict(last_row)[0]
            predictions.append(pred_close)
            
            # Update features for next step
            # Shift lags
            last_row['Close_Lag_5'] = last_row['Close_Lag_3']
            last_row['Close_Lag_3'] = last_row['Close_Lag_2']
            last_row['Close_Lag_2'] = last_row['Close_Lag_1']
            last_row['Close_Lag_1'] = current_close
            
            # Update Close
            last_row['Close'] = pred_close
            
            # Update Returns (approximate)
            last_row['Returns'] = (pred_close - current_close) / current_close
            
            # Update Range (assume average range of last 5 days)
            last_row['Range'] = df['High'].iloc[-5:] - df['Low'].iloc[-5:].mean()
            
            current_close = pred_close
            
        return predictions
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# 6. Screener Feature
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_screener_data():
    """
    인기 종목 (KRX/US) 실시간 시세 조회 및 스크린샷 기반 데이터 구성
    """
    # 환율 조회 (USD -> KRW)
    try:
        usd_krw = yf.Ticker("USDKRW=X").history(period="1d")['Close'].iloc[-1]
    except:
        usd_krw = 1400.0 # Fallback

    # Predefined Popular Stocks
    stocks = {
        'KRX': [
            {'ticker': '005930.KS', 'name': '삼성전자', 'category': '반도체'},
            {'ticker': '000660.KS', 'name': 'SK하이닉스', 'category': '반도체'},
            {'ticker': '373220.KS', 'name': 'LG에너지솔루션', 'category': '2차전지'},
            {'ticker': '207940.KS', 'name': '삼성바이오로직스', 'category': '바이오'},
            {'ticker': '005380.KS', 'name': '현대차', 'category': '자동차'},
            {'ticker': '000270.KS', 'name': '기아', 'category': '자동차'},
            {'ticker': '005490.KS', 'name': 'POSCO홀딩스', 'category': '철강/소재'},
            {'ticker': '035420.KS', 'name': 'NAVER', 'category': '인터넷'},
            {'ticker': '035720.KS', 'name': '카카오', 'category': '인터넷'},
            {'ticker': '068270.KS', 'name': '셀트리온', 'category': '바이오'},
            {'ticker': '105560.KS', 'name': 'KB금융', 'category': '금융'},
            {'ticker': '055550.KS', 'name': '신한지주', 'category': '금융'},
            {'ticker': '006400.KS', 'name': '삼성SDI', 'category': '2차전지'},
            {'ticker': '051910.KS', 'name': 'LG화학', 'category': '화학'},
            {'ticker': '012330.KS', 'name': '현대모비스', 'category': '자동차부품'},
        ],
        'US': [
            {'ticker': 'AAPL', 'name': '애플', 'category': '기술/하드웨어'},
            {'ticker': 'MSFT', 'name': '마이크로소프트', 'category': '소프트웨어'},
            {'ticker': 'NVDA', 'name': '엔비디아', 'category': '반도체'},
            {'ticker': 'AMZN', 'name': '아마존', 'category': '유통/클라우드'},
            {'ticker': 'GOOGL', 'name': '구글 (알파벳)', 'category': '인터넷'},
            {'ticker': 'META', 'name': '메타 (페이스북)', 'category': '인터넷'},
            {'ticker': 'TSLA', 'name': '테슬라', 'category': '자동차'},
            {'ticker': 'TSM', 'name': 'TSMC', 'category': '반도체'},
            {'ticker': 'AVGO', 'name': '브로드컴', 'category': '반도체'},
            {'ticker': 'LLY', 'name': '일라이 릴리', 'category': '제약'},
            {'ticker': 'JPM', 'name': 'JP모건 체이스', 'category': '금융'},
            {'ticker': 'V', 'name': '비자', 'category': '금융'},
            {'ticker': 'WMT', 'name': '월마트', 'category': '유통'},
            {'ticker': 'XOM', 'name': '엑슨모빌', 'category': '에너지'},
            {'ticker': 'UNH', 'name': '유나이티드헬스', 'category': '헬스케어'},
            {'ticker': 'MA', 'name': '마스터카드', 'category': '금융'},
            {'ticker': 'PG', 'name': 'P&G (프록터앤갬블)', 'category': '소비재'},
            {'ticker': 'COST', 'name': '코스트코', 'category': '유통'},
            {'ticker': 'JNJ', 'name': '존슨앤존슨', 'category': '헬스케어'},
            {'ticker': 'HD', 'name': '홈디포', 'category': '유통'},
        ]
    }
    
    results = []
    
    # Batch Fetch for efficiency (per market)
    try:
        # Flatten tickers
        all_tickers = [s['ticker'] for s in stocks['KRX']] + [s['ticker'] for s in stocks['US']]
        
        # yfinance batch download
        batch_data = yf.download(all_tickers, period="5d", group_by='ticker', threads=False, progress=False)
        
        # Check if batch_data is MultiIndex
        is_multi = isinstance(batch_data.columns, pd.MultiIndex)
        
        for market_type, stock_list in stocks.items():
            for s in stock_list:
                t = s['ticker']
                try:
                    # Extract Data
                    if is_multi:
                        if t not in batch_data.columns.levels[0]:
                            continue
                        df_t = batch_data[t]
                    else:
                        if len(all_tickers) == 1:
                            df_t = batch_data
                        else:
                            continue # Should not happen
                            
                    if df_t.empty or len(df_t) < 2:
                        continue
                        
                    # Calculate Metrics
                    last_close = df_t['Close'].iloc[-1]
                    prev_close = df_t['Close'].iloc[-2]
                    change = last_close - prev_close
                    pct_change = (change / prev_close) * 100
                    volume = df_t['Volume'].iloc[-1]
                    
                    # Market Cap
                    mkt_cap = 0
                    try:
                        # Try to get market cap from fast_info (requires separate Ticker object)
                        # To optimize, we could create Ticker objects in batch or cache this
                        # For now, we fetch it individually as it's the only way with current yfinance structure
                        # Note: This might slow down the screener slightly (35 requests)
                        t_obj = yf.Ticker(t)
                        mkt_cap = t_obj.fast_info.get('marketCap', 0)
                    except:
                        pass
                    
                    # 3. Format Data (Like Screenshot)
                    display_price = last_close
                    display_mkt_cap_str = "-"
                    
                    if market_type == 'US':
                        # Convert to KRW for display? Screenshot showed mixed.
                        # User screenshot 2 showed "12,692원" for VNDA (US Stock).
                        # So yes, convert to KRW.
                        display_price_krw = last_close * usd_krw
                        price_str = f"{display_price_krw:,.0f}원"
                        mkt_cap_krw = mkt_cap * usd_krw
                    else:
                        price_str = f"{last_close:,.0f}원"
                        mkt_cap_krw = mkt_cap
                        
                    # Format Market Cap
                    if mkt_cap_krw > 0:
                        if mkt_cap_krw > 1_000_000_000_000: # 1조 이상
                            display_mkt_cap_str = f"{mkt_cap_krw/1_000_000_000_000:.1f}조원"
                        elif mkt_cap_krw > 100_000_000: # 1억 이상
                            display_mkt_cap_str = f"{mkt_cap_krw/100_000_000:.1f}억원"
                        
                    # Analyst Rating (Dummy)
                    analyst = "강력 매수" if pct_change > 3 else "매수" if pct_change > 0 else "중립" if pct_change > -3 else "매도"
                    
                    results.append({
                        'Market': '국내' if market_type == 'KRX' else '해외',
                        'Name': f"{s['name']}\n{t}", 
                        'Price_Raw': display_price,
                        'Price': price_str,
                        'Change': pct_change / 100, 
                        'Category': s['category'],
                        'Market_Cap': display_mkt_cap_str,
                        'Volume': volume,
                        'Analyst': analyst
                    })
                except Exception as e:
                    continue
    except Exception as e:
        return pd.DataFrame()
        
    # Sort by Change (Desc) to mimic "Top Gainers"
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by='Change', ascending=False).reset_index(drop=True)
        df_res.index += 1 # Rank starts at 1
        df_res.reset_index(inplace=True)
        df_res.rename(columns={'index': 'Rank'}, inplace=True)
        
    return df_res

def render_screener_page():
    st.title("📋 주식 골라보기 (실시간 인기 종목)")
    st.caption("국내(KRX) 및 해외(US) 주요 종목의 실시간 시세와 정보를 확인하세요.")
    
    with st.spinner('실시간 데이터 조회 중... (환율 변환 포함)'):
        df = fetch_screener_data()
    
    if df.empty:
        st.warning("데이터를 불러올 수 없습니다.")
        return

    # Custom CSS for Table
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] {
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(["🇰🇷 국내 (Domestic)", "🇺🇸 해외 (Overseas)"])
    
    # Column Configuration
    column_config = {
        "Rank": st.column_config.NumberColumn("순위", format="%d"),
        "Name": st.column_config.TextColumn("종목명", help="티커 포함"),
        "Price": st.column_config.TextColumn("현재가"),
        "Change": st.column_config.NumberColumn(
            "등락률",
            format="%.2f%%",
        ),
        "Category": "카테고리",
        "Market_Cap": "시가총액",
        "Volume": st.column_config.NumberColumn("거래량", format="%d주"),
        "Analyst": "애널리스트 분석",
    }
    
    cols = ["Rank", "Name", "Price", "Change", "Category", "Market_Cap", "Volume", "Analyst"]

    with tab1:
        # Filter KRX
        df_krx = df[df['Market'] == '국내'].reset_index(drop=True)
        df_krx['Rank'] = df_krx.index + 1
        st.dataframe(
            df_krx[cols],
            column_config=column_config,
            hide_index=True,
            use_container_width=True
        )
        
    with tab2:
        # Filter US
        df_us = df[df['Market'] == '해외'].reset_index(drop=True)
        df_us['Rank'] = df_us.index + 1
        st.dataframe(
            df_us[cols],
            column_config=column_config,
            hide_index=True,
            use_container_width=True
        )



# -----------------------------------------------------------------------------
# 7. Render Pages
# -----------------------------------------------------------------------------
def render_analysis_page():
    # Sidebar: Single Analysis Mode
    market = st.sidebar.radio("시장 선택", ["KRX (한국)", "US (미국)"])
    
    # Market Sentiment Widget
    with st.sidebar:
        st.markdown("### 🌍 시장 심리 (Sentiment)")
        m_sent = fetch_market_sentiment(market)
        if m_sent:
            color = "red" if m_sent['change'] < 0 else "green" # US style
            if market == "KRX (한국)": color = "red" if m_sent['change'] > 0 else "blue" # KRX style
            
            st.metric(m_sent['name'], f"{m_sent['value']:,.2f}", f"{m_sent['change']:.2f}%")
            st.caption(f"상태: {m_sent['sentiment']}")
        else:
            st.caption("시장 데이터 로딩 실패")
            
    st.sidebar.markdown("---")
    
    # Input Section
    default_ticker = "삼성전자" if market == "KRX (한국)" else "애플"
    user_input = st.sidebar.text_input("종목명 또는 코드 입력", value=default_ticker, help="예: 삼성전자, 애플, 005930, TSLA")
    
    period = st.sidebar.selectbox("분석 기간", ["1y", "2y", "5y"], index=1)
    
    if st.sidebar.button("분석 시작", type="primary"):
        # Resolve Ticker
        resolved_ticker, resolved_market, company_name = resolve_ticker(user_input)
        
        if not resolved_ticker:
            st.error(f"'{user_input}'을(를) 찾을 수 없습니다. 정확한 종목명이나 코드를 입력해주세요.")
            return
            
        # Check Market Mismatch (Optional warning, but we can auto-correct market)
        # If user selected KRX but typed "Apple" -> resolved_market="US"
        # We should use the resolved market.
        if resolved_market == "KRX":
            market_label = "KRX (한국)"
        else:
            market_label = "US (미국)"
            
        if market_label != market:
             st.info(f"'{company_name}'은(는) {resolved_market} 시장 종목입니다. 시장 설정을 자동으로 변경합니다.")
             market = market_label

        st.success(f"분석 대상: {company_name} ({resolved_ticker})")

        with st.spinner(f'{company_name} ({resolved_ticker}) 데이터 분석 중...'):
            # 1. Fetch Basic Data (yfinance)
            df, news, symbol = fetch_stock_data(resolved_ticker, market, period=period)
            
            if df is None:
                st.error(symbol) # Error message
                return
                
            # 2. Fetch Naver Data (KRX Only)
            naver_data = None
            if market == "KRX (한국)":
                naver_data = fetch_naver_finance_data(symbol)
                
            # ---------------------------------------------------------
            # UI Layout: Header & Summary
            # ---------------------------------------------------------
            last_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            change = last_close - prev_close
            pct_change = (change / prev_close) * 100
            
            # Currency Symbol
            currency = "원" if market == "KRX (한국)" else "$"
            price_fmt = f"{last_close:,.0f}" if market == "KRX (한국)" else f"{last_close:.2f}"
            
            st.title(f"{symbol} AI 분석 리포트")
            st.markdown(f"**기준일:** {datetime.datetime.now().strftime('%Y-%m-%d')} | **시장:** {market}")
            
            # Top Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("현재가", f"{price_fmt}{currency}", f"{pct_change:+.2f}%")
            col2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
            col3.metric("거래량", f"{df['Volume'].iloc[-1]:,.0f}")
            col4.metric("ATR (변동성)", f"{df['ATR'].iloc[-1]:,.2f}")
            
            # KRX Fundamentals (Naver)
            if market == "KRX (한국)" and naver_data:
                st.markdown("### 🏢 기업 펀더멘털 (Naver 금융)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("시가총액", naver_data.get('market_cap', '-'))
                c2.metric("PER", naver_data.get('per', '-'))
                c3.metric("PBR", naver_data.get('pbr', '-'))
                c4.metric("투자의견", naver_data.get('opinion', '-'))
            
            st.markdown("---")
            
            # ---------------------------------------------------------
            # Tabs: Organized Analysis
            # ---------------------------------------------------------
            tab1, tab2, tab3, tab4 = st.tabs(["📊 차트 분석", "🧠 AI 진단", "🔮 미래 예측", "📰 뉴스/공시"])
            
            # --- Tab 1: Chart ---
            with tab1:
                st.subheader("📈 종합 차트 (Price + MA + BB)")
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, row_heights=[0.7, 0.3])

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                                
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='MA 60'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='BB Lower'), row=1, col=1)
                
                colors = ['blue' if row['Open'] - row['Close'] >= 0 else 'red' for index, row in df.iterrows()]
                if market == "US (미국)":
                    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
                    
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
                
                fig.update_layout(height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
            # --- Tab 2: AI Diagnosis ---
            with tab2:
                st.subheader("🧠 AI 기술적 분석 (Chain-of-Thought)")
                
                score, steps, talib_patterns = analyze_trend_score(df)
                
                col_a1, col_a2 = st.columns([1, 1])
                
                with col_a1:
                    st.markdown(f"### 🏆 종합 기술적 점수: **{score}점** / 100점")
                    st.progress(score / 100)
                    
                    # Chart Patterns (Geometric)
                    analyzer = ChartPatternAnalyzer(df)
                    geo_patterns = analyzer.detect_patterns()
                    
                    st.markdown("#### 📐 기하학적 차트 패턴")
                    if geo_patterns:
                        for p in geo_patterns:
                            st.info(f"**{p['name']}**: {p['desc']} ({p['signal']})")
                    else:
                        st.caption("특이한 기하학적 패턴이 감지되지 않았습니다.")

                    # TA-Lib Patterns
                    st.markdown("#### 🕯️ 캔들스틱 패턴 (TA-Lib)")
                    if talib_patterns:
                        for p in talib_patterns:
                            icon = "📈" if p['direction'] == '상승' else "📉" if p['direction'] == '하락' else "➖"
                            if p['direction'] == '상승':
                                st.success(f"{icon} **{p['name']}**: {p['desc']} (신뢰도: {p['conf']}%)")
                            elif p['direction'] == '하락':
                                st.error(f"{icon} **{p['name']}**: {p['desc']} (신뢰도: {p['conf']}%)")
                            else:
                                st.info(f"{icon} **{p['name']}**: {p['desc']} (신뢰도: {p['conf']}%)")
                    else:
                        st.caption("특이한 캔들 패턴이 감지되지 않았습니다.")
                        if not TALIB_AVAILABLE:
                            st.warning("TA-Lib 라이브러리가 설치되지 않았습니다.")

                with col_a2:
                    st.markdown("#### 📝 단계별 분석 리포트")
                    for step in steps:
                        with st.expander(step['step'], expanded=True):
                            st.write(step['result'])
                            st.caption(f"Score Impact: {step['score']:+d}")

            # --- Tab 3: Forecast & Risk ---
            with tab3:
                st.subheader("🔮 미래 가격 예측 (AI Ensemble)")
                
                col_f1, col_f2 = st.columns([2, 1])
                forecast_days = 30
                expected_price = 0
                
                with col_f1:
                    if PROPHET_AVAILABLE:
                        df_p = df.reset_index()[['Date', 'Close']]
                        df_p.columns = ['ds', 'y']
                        df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                        
                        m = Prophet(daily_seasonality=True)
                        m.fit(df_p)
                        future = m.make_future_dataframe(periods=forecast_days)
                        forecast = m.predict(future)
                        
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name='Actual', line=dict(color='black')))
                        fig_p.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prophet Predict', line=dict(color='blue')))
                        fig_p.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(width=0), showlegend=False))
                        fig_p.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', showlegend=False))
                        st.plotly_chart(fig_p, use_container_width=True)
                        
                        expected_price = forecast['yhat'].iloc[-1]
                    else:
                        st.warning("Prophet 미설치로 선형 회귀 대체")
                        # Linear Regression logic (simplified for brevity)
                        X = np.arange(len(df)).reshape(-1, 1)
                        y = df['Close'].values
                        reg = LinearRegression().fit(X, y)
                        future_X = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
                        future_pred = reg.predict(future_X)
                        expected_price = future_pred[-1]
                        
                        fig_l = go.Figure()
                        fig_l.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual'))
                        # Create dates... (omitted for brevity, assume linear plot)
                        st.plotly_chart(fig_l, use_container_width=True)

                with col_f2:
                    st.markdown("#### 예측 요약")
                    
                    if XGBOOST_AVAILABLE and PROPHET_AVAILABLE:
                        xgb_preds = run_xgboost_forecast(df, forecast_days)
                        if xgb_preds:
                            xgb_final = xgb_preds[-1]
                            ensemble_price = (expected_price * 0.6) + (xgb_final * 0.4)
                            ensemble_change = (ensemble_price - last_close) / last_close * 100
                            
                            st.metric("AI 앙상블 예상가 (30일 후)", f"{ensemble_price:,.0f}{currency}", f"{ensemble_change:+.2f}%")
                            
                            # Peak Detection
                            prophet_preds = forecast['yhat'].tail(forecast_days).values
                            ensemble_preds = (prophet_preds * 0.6) + (np.array(xgb_preds) * 0.4)
                            peak_idx = np.argmax(ensemble_preds)
                            peak_date = forecast['ds'].tail(forecast_days).iloc[peak_idx]
                            peak_price = ensemble_preds[peak_idx]
                            
                            st.info(f"📅 추천 매도 타이밍: **{peak_date.strftime('%Y-%m-%d')}**")
                            st.caption(f"예상 최고가: {peak_price:,.0f}{currency}")
                            
                            with st.expander("앙상블 차트 보기"):
                                fig_e = go.Figure()
                                fig_e.add_trace(go.Scatter(x=forecast['ds'].tail(forecast_days), y=prophet_preds, name='Prophet'))
                                fig_e.add_trace(go.Scatter(x=forecast['ds'].tail(forecast_days), y=xgb_preds, name='XGBoost'))
                                fig_e.add_trace(go.Scatter(x=forecast['ds'].tail(forecast_days), y=ensemble_preds, name='Ensemble', line=dict(width=3)))
                                st.plotly_chart(fig_e, use_container_width=True)
                    else:
                        st.metric("30일 후 예상가", f"{expected_price:,.0f}{currency}")

                st.markdown("---")
                st.subheader("🛡️ 리스크 관리 (ATR 기반)")
                scenarios = calculate_risk_scenarios(last_close, df['ATR'].iloc[-1])
                
                # Use 3 columns for cards
                cols = st.columns(3)
                
                for idx, (strategy_name, data) in enumerate(scenarios.items()):
                    with cols[idx]:
                        html = f"""
                        <div class="risk-card" style="background-color: {data['color']};">
                            <div class="risk-title">{data['icon']} {strategy_name}</div>
                            <div class="risk-desc">{data['desc']}</div>
                            <div class="risk-price">
                                <span class="risk-label">🎯 목표가</span>
                                <span style="color: #d32f2f; font-weight:bold;">{data['target']:,.0f} {currency}</span>
                            </div>
                            <div class="risk-price">
                                <span class="risk-label">🛑 손절가</span>
                                <span style="color: #1976d2; font-weight:bold;">{data['stop']:,.0f} {currency}</span>
                            </div>
                            <div style="margin-top: 10px; font-size: 12px; text-align: right; color: #555;">
                                ⚖️ 손익비 {data['ratio']}
                            </div>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)

            # --- Tab 4: News & Disclosures ---
            with tab4:
                if market == "KRX (한국)" and naver_data:
                    c_n1, c_n2 = st.columns(2)
                    with c_n1:
                        st.subheader("📰 주요 뉴스")
                        if naver_data['news']:
                            for n in naver_data['news']:
                                st.markdown(f"- [{n['title']}]({n['link']})")
                        else: st.write("뉴스 없음")
                        
                    with c_n2:
                        st.subheader("� 최근 공시")
                        if naver_data['disclosures']:
                            for d in naver_data['disclosures']:
                                st.markdown(f"- [{d['title']}]({d['link']})")
                        else: st.write("공시 없음")
                        
                    st.divider()
                    st.caption("출처: 네이버 금융")
                    
                else:
                    st.subheader("📰 관련 뉴스 (Google RSS)")
                    if news:
                        for item in news:
                            st.markdown(f"- [{item['title']}]({item['link']}) - *{item['publisher']}*")
                    else:
                        st.info("검색된 뉴스가 없습니다.")

            st.markdown("---")
            st.caption("⚠️ 본 분석 리포트는 AI 모델 및 기술적 지표에 기반한 참고용 자료이며, 투자에 대한 책임은 본인에게 있습니다.")

# -----------------------------------------------------------------------------
# 8. Main Application Entry
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("📈 주식 AI 예측 시스템")
    
    # Navigation
    page = st.sidebar.radio("메뉴 선택", ["🔍 종목 상세 분석", "📋 주식 골라보기 (Screener)"])
    st.sidebar.markdown("---")
    
    if page == "📋 주식 골라보기 (Screener)":
        render_screener_page()
    else:
        render_analysis_page()

if __name__ == "__main__":
    main()
