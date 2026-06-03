"""데이터 수집 레이어 — Naver Finance 스크래핑 + yfinance + Upbit API.

출처: k-ant-daily/scripts/fetch_news.py 에서 핵심 함수 이식.
StockOracle용으로 독립 함수 형태로 재구성 (파일 I/O 없음, 순수 반환값).

주요 함수:
  fetch_macro_context()       — 거시 지표 통합 수집 (지수 + FX + 간밤 + 뉴스)
  fetch_stock_snapshot()      — 단일 종목 스냅샷 (시세 + 뉴스 + 히스토리 + 오버나이트)
  fetch_stock_list_snapshot() — 종목 리스트 일괄 수집

[성능 최적화 내역]
  - fetch_overnight_markets() : yf.download() 단일 배치 (9회→1회 HTTP 요청)
  - fetch_macro_context()     : ThreadPoolExecutor 5-way 병렬 수집
  - 모듈 레벨 TTL 캐시         : 장중 3분, 장 외 10분 (인스턴스 생존 시 재사용)
  - HTTP timeout              : 15s → 7s (에러 대기 절감)
"""
from __future__ import annotations

import re
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from threading import Lock as _Lock
from typing import Any

import requests
from bs4 import BeautifulSoup

KST = timezone(timedelta(hours=9))

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# ── 모듈 레벨 TTL 캐시 (Vercel 웜 인스턴스 재사용) ───────────────────────────
# 장중(09:00~15:30 KST): 3분, 장 외: 10분
_MACRO_CACHE:  dict = {"data": None, "ts": 0.0}
_MACRO_LOCK    = _Lock()   # Thundering-herd 방지 (동시 다중 요청 중복 수집 차단)

_SECTOR_CACHE: dict = {"data": None, "ts": 0.0}
_SECTOR_LOCK   = _Lock()

def _macro_ttl_seconds() -> int:
    """현재 KST 기준 시장 상태에 따른 캐시 TTL 반환.

    장중(09:00~15:30): 3분  — 지수·환율이 분 단위로 변동
    미국 정규장(22:30~05:00): 5분 — 간밤 지표 변동 반영
    장 외(야간·주말): 10분  — 데이터 거의 고정, 서버 부하 최소화
    """
    now_kst = datetime.now(KST)
    wday = now_kst.weekday()   # 0=월 … 6=일
    h    = now_kst.hour + now_kst.minute / 60.0

    if wday >= 5:              # 주말
        return 600             # 10분
    if 9.0 <= h < 15.5:       # 국내 정규장
        return 180             # 3분
    if 22.5 <= h or h < 5.0:  # 미국 정규장(EDT 기준, EST는 23.5)
        return 300             # 5분
    return 600                 # 장 외 (오전 장전·오후 장후·심야)


def _sector_ttl_seconds() -> int:
    """섹터 흐름 캐시 TTL: 장중 3분, 장외 10분."""
    now_kst = datetime.now(KST)
    wday = now_kst.weekday()
    h    = now_kst.hour + now_kst.minute / 60.0
    if wday >= 5:               return 600   # 주말
    if 9.0 <= h < 15.5:        return 180   # 국내 장중
    return 600                               # 장외


# ── 내부 HTTP 헬퍼 ──────────────────────────────────────────────────────────

def _get(url: str, referer: str | None = None) -> BeautifulSoup:
    headers = {"User-Agent": _UA, "Accept-Language": "ko-KR,ko;q=0.9"}
    if referer:
        headers["Referer"] = referer
    r = requests.get(url, headers=headers, timeout=7)   # 15s → 7s
    r.raise_for_status()
    ct = r.headers.get("content-type", "").lower()
    if "charset" not in ct:
        r.encoding = "euc-kr"
    return BeautifulSoup(r.text, "html.parser")


def _abs_naver_link(href: str) -> str:
    if "news_read" in href:
        office  = re.search(r"office_id=(\d+)", href)
        article = re.search(r"article_id=(\d+)", href)
        if office and article:
            return f"https://n.news.naver.com/mnews/article/{office.group(1)}/{article.group(1)}"
    if href.startswith("/"):
        return "https://finance.naver.com" + href
    return href


# ── 거시 지표 ────────────────────────────────────────────────────────────────

def fetch_market_indices() -> dict[str, Any]:
    """KOSPI / KOSDAQ / KOSPI200 현재값 + 등락.

    반환 예:
      {"KOSPI": {"value":"2,600.12","change_abs":"+12.34","change_pct":"+0.48%","direction":"up"}, ...}
    """
    soup = _get("https://finance.naver.com/sise/")
    result: dict = {}
    for name, val_sel, chg_sel in (
        ("KOSPI",    "#KOSPI_now",  "#KOSPI_change"),
        ("KOSDAQ",   "#KOSDAQ_now", "#KOSDAQ_change"),
        ("KOSPI200", "#KPI200_now", "#KPI200_change"),
    ):
        val_el = soup.select_one(val_sel)
        chg_el = soup.select_one(chg_sel)
        if not val_el:
            continue
        value     = val_el.get_text(" ", strip=True)
        raw_chg   = chg_el.get_text(" ", strip=True) if chg_el else ""
        parsed    = _parse_index_change(raw_chg)
        result[name] = {
            "value":      value,
            "change_abs": parsed["abs"],
            "change_pct": parsed["pct"],
            "direction":  parsed["direction"],
        }
    return result


def _parse_index_change(raw: str) -> dict:
    parts = raw.split()
    if len(parts) < 3:
        return {"abs": "", "pct": "", "direction": "flat"}
    raw_abs, raw_pct, rise_fall = parts[0], parts[1], parts[2]
    if rise_fall == "상승":
        direction, sign = "up", "+"
    elif rise_fall == "하락":
        direction, sign = "down", "-"
    else:
        direction, sign = "flat", ""
    pct_clean = raw_pct.lstrip("+-")
    return {
        "abs": f"{sign}{raw_abs}" if sign else raw_abs,
        "pct": f"{sign}{pct_clean}" if sign else pct_clean,
        "direction": direction,
    }


def fetch_fx() -> list[dict]:
    """주요 환율 (USD/KRW 등 상위 4개)."""
    soup = _get("https://finance.naver.com/marketindex/")
    rates = []
    for li in soup.select("ul#exchangeList li")[:4]:
        name  = li.select_one("h3.h_lst span.blind")
        value = li.select_one("span.value")
        chg   = li.select_one("span.change")
        if name and value:
            rates.append({
                "name":   name.get_text(strip=True),
                "value":  value.get_text(strip=True),
                "change": chg.get_text(strip=True) if chg else "",
            })
    return rates


def fetch_macro_news(limit: int = 15) -> list[dict]:
    """네이버 금융 메인 뉴스 (거시/시장 전반)."""
    soup = _get("https://finance.naver.com/news/mainnews.naver")
    items: list[dict] = []
    for li in soup.select("li.block1"):
        subj = li.select_one("dd.articleSubject a") or li.select_one("dt a")
        summ = li.select_one("dd.articleSummary")
        if not subj:
            continue
        href   = _abs_naver_link(subj.get("href", ""))
        source = date = summary_text = ""
        if summ:
            press = summ.select_one(".press")
            wdate = summ.select_one(".wdate")
            source = press.get_text(strip=True) if press else ""
            date   = wdate.get_text(strip=True) if wdate else ""
            for s in summ.select(".press, .wdate, .source"):
                s.extract()
            summary_text = summ.get_text(" ", strip=True)
        items.append({
            "title":   subj.get_text(strip=True),
            "link":    href,
            "source":  source,
            "date":    date,
            "summary": summary_text,
        })
        if len(items) >= limit:
            break
    return items


def fetch_overnight_markets() -> list[dict]:
    """간밤 해외 시장 — yf.download() 단일 배치로 9종목 동시 수집.

    기존: yf.Ticker(sym).history() × 9회 → 18~27s
    변경: yf.download([...9개...])  × 1회 → 2~3s
    """
    import yfinance as yf

    TICKERS = [
        ("^GSPC",    "S&P 500"),
        ("^DJI",     "다우존스"),
        ("^IXIC",    "나스닥"),
        ("^VIX",     "VIX"),
        ("^KS200",   "KOSPI200 (종가)"),
        ("CL=F",     "WTI 원유"),
        ("GC=F",     "금"),
        ("BTC-USD",  "비트코인"),
        ("DX-Y.NYB", "달러인덱스"),
    ]
    syms   = [s for s, _ in TICKERS]
    labels = {s: l for s, l in TICKERS}

    try:
        # 9종목 단일 배치 요청 — group_by="ticker" 로 df[sym]["Close"] 접근
        df = yf.download(
            syms,
            period="5d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        print(f"[warn] yf.download overnight batch: {e}", file=sys.stderr)
        return []

    out: list[dict] = []
    for sym in syms:
        try:
            # MultiIndex: df[sym]["Close"], 단일 티커면 df["Close"]
            closes = (df[sym]["Close"] if sym in df.columns.get_level_values(0)
                      else df["Close"]).dropna()
            if len(closes) < 2:
                continue
            prev = float(closes.iloc[-2])
            last = float(closes.iloc[-1])
            diff = last - prev
            pct  = (diff / prev * 100) if prev else 0.0
            out.append({
                "symbol":     sym,
                "name":       labels[sym],
                "value":      f"{last:,.2f}",
                "change":     f"{'+'if pct>=0 else ''}{pct:.2f}%",
                "change_abs": f"{'+'if diff>=0 else '-'}{abs(diff):,.2f}",
                "change_pct": round(pct, 2),
                "direction":  "up" if pct > 0 else ("down" if pct < 0 else "flat"),
                "as_of":      str(closes.index[-1].date()),
            })
        except Exception as e:
            print(f"[warn] overnight parse {sym}: {e}", file=sys.stderr)
    return out


def fetch_upbit_crypto(markets: list[str] | None = None) -> dict[str, Any]:
    """업비트 KRW 암호화폐 시세 (BTC, ETH)."""
    if markets is None:
        markets = ["KRW-BTC", "KRW-ETH"]
    try:
        r = requests.get(
            "https://api.upbit.com/v1/ticker",
            params={"markets": ",".join(markets)},
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"[warn] upbit: {e}", file=sys.stderr)
        return {}

    out: dict = {}
    for d in r.json():
        market    = d.get("market")
        direction = {"RISE": "up", "FALL": "down"}.get(d.get("change"), "flat")
        price     = float(d.get("trade_price") or 0)
        abs_chg   = float(d.get("signed_change_price") or 0)
        pct       = float(d.get("signed_change_rate") or 0) * 100
        out[market] = {
            "value":      f"{round(price):,}",
            "change_abs": f"{'+'if abs_chg>0 else '-' if abs_chg<0 else ''}{abs(round(abs_chg)):,}",
            "change_pct": f"{'+'if pct>0 else ''}{pct:.2f}%",
            "direction":  direction,
        }
    return out


# ── 개별 종목 ────────────────────────────────────────────────────────────────

def _fetch_quote_mobile_json(code: str) -> dict | None:
    """네이버 모바일 JSON API로 시세 수집.

    HTML 스크래핑 대비 응답 크기·처리 시간 대폭 절감.
    실패 시 None 반환 → HTML 폴백.
    """
    try:
        url  = f"https://m.stock.naver.com/api/stock/{code}/basic"
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Mobile Safari/537.36",
            "Referer":    "https://m.stock.naver.com/",
            "Accept":     "application/json",
        }
        r = requests.get(url, headers=hdrs, timeout=5)
        if r.status_code != 200:
            return None
        d = r.json()

        price_str = (d.get("closePrice") or "").replace(",", "")
        diff_str  = (d.get("compareToPreviousClosePrice") or "0").replace(",", "")
        vol_str   = (d.get("tradeVolume") or "").replace(",", "")
        # compareToPreviousPrice.code: "5" = 하락, "2" = 상승, 그 외 = 보합
        code_type = (d.get("compareToPreviousPrice") or {}).get("code", "3")

        price_num = float(price_str)
        diff_num  = float(diff_str)
        if code_type == "5":           # 하락 → diff는 절댓값이므로 음수 처리
            diff_num = -diff_num
        prev_num  = price_num - diff_num
        pct_num   = (diff_num / prev_num * 100) if prev_num else 0.0
        direction = "up" if diff_num > 0 else ("down" if diff_num < 0 else "flat")
        sign      = "+" if diff_num > 0 else ""

        out: dict = {
            "price":          f"{int(price_num):,}",
            "change":         f"{abs(int(diff_num)):,}",
            "change_pct":     f"{sign}{pct_num:.2f}%",
            "change_pct_num": round(pct_num, 2),
            "direction":      direction,
        }
        if vol_str:
            try:
                out["volume"] = int(vol_str)
            except ValueError:
                pass
        return out
    except Exception:
        return None


def fetch_stock_quote(code: str) -> dict:
    """네이버 금융 현재 시세 — 모바일 JSON API 우선, HTML 폴백."""
    # ① 모바일 JSON API (빠름 — 페이지당 ~50ms vs HTML ~300ms)
    result = _fetch_quote_mobile_json(code)
    if result:
        return result

    # ② HTML 폴백 (모바일 API 실패 시)
    out: dict = {}
    try:
        url  = f"https://finance.naver.com/item/main.naver?code={code}"
        soup = _get(url)
        today = soup.select_one("div.today")
        if not today:
            return out
        now_el = today.select_one("p.no_today .blind")
        if now_el:
            out["price"] = now_el.get_text(strip=True)
        exday = today.select_one("p.no_exday")
        if exday:
            blinds    = [b.get_text(strip=True) for b in exday.select("em .blind, span .blind")]
            em        = exday.select_one("em")
            direction = ""
            if em:
                cls = " ".join(em.get("class") or [])
                direction = "up" if "up" in cls else ("down" if "down" in cls else "")
            if blinds:
                out["change"] = blinds[0] if len(blinds) >= 1 else ""
                pct = blinds[1] if len(blinds) >= 2 else ""
                if pct:
                    sign = "-" if direction == "down" else "+" if direction == "up" else ""
                    out["change_pct"] = f"{sign}{pct}"
                    try:
                        out["change_pct_num"] = float(pct.replace("%", "")) * (-1 if direction == "down" else 1)
                    except ValueError:
                        pass
                out["direction"] = direction
        for th in soup.select("table.rwidth th, table.lwidth th, th"):
            if th.get_text(strip=True) == "거래량":
                td = th.find_next("em") or th.find_next("td")
                if td:
                    try:
                        out["volume"] = int(td.get_text(strip=True).replace(",", ""))
                    except ValueError:
                        pass
                break
    except Exception:
        pass
    return out


def fetch_stock_news(code: str, limit: int = 10) -> list[dict]:
    """네이버 금융 종목 뉴스 (최근 기사)."""
    url     = f"https://finance.naver.com/item/news_news.naver?code={code}&page=1"
    referer = f"https://finance.naver.com/item/main.naver?code={code}"
    soup    = _get(url, referer=referer)
    items: list[dict] = []
    for tr in soup.select("table.type5 tbody tr"):
        if "relation_lst" in (tr.get("class") or []):
            continue
        title_a  = tr.select_one("td.title a.tit")
        info_td  = tr.select_one("td.info")
        date_td  = tr.select_one("td.date")
        if not title_a:
            continue
        items.append({
            "title":  title_a.get_text(strip=True),
            "link":   _abs_naver_link(title_a.get("href", "")),
            "source": info_td.get_text(strip=True) if info_td else "",
            "date":   date_td.get_text(strip=True) if date_td else "",
        })
        if len(items) >= limit:
            break
    return items


def fetch_stock_disclosures(code: str, limit: int = 5) -> list[dict]:
    """네이버 금융 공시 목록."""
    url     = f"https://finance.naver.com/item/news_notice.naver?code={code}"
    referer = f"https://finance.naver.com/item/main.naver?code={code}"
    soup    = _get(url, referer=referer)
    items: list[dict] = []
    for td in soup.select("td.title"):
        a = td.select_one("a.tit")
        if not a:
            continue
        date_td = td.find_next_sibling("td", class_="date")
        href    = a.get("href", "")
        if href.startswith("/"):
            href = "https://finance.naver.com" + href
        items.append({
            "title": a.get_text(strip=True),
            "link":  href,
            "date":  date_td.get_text(strip=True) if date_td else "",
        })
        if len(items) >= limit:
            break
    return items


def fetch_dart_disclosures(code: str, limit: int = 5,
                           api_key: str | None = None,
                           corp_code: str | None = None) -> list[dict]:
    """DART 전자공시 OpenAPI 최근 공시 목록.

    공식 전자공시(DART) 원천. 키/corp_code가 설정된 경우에만 동작하며,
    미설정 시 []를 반환(이 경우 네이버 공시 미러가 대체 제공).

    설정:
      · 환경변수 DART_API_KEY            — OpenDART 인증키
      · corp_code(8자리 DART 고유번호)   — 인자 또는 환경변수
        DART_CORP_CODE_MAP(JSON, 예: '{"005930":"00126380"}') 로 매핑

    반환: [{title, link, date, source_type:"dart"}]
    """
    import os
    import json as _json

    key = api_key or os.environ.get("DART_API_KEY")
    if not key:
        return []
    cc = corp_code
    if not cc:
        raw = os.environ.get("DART_CORP_CODE_MAP")
        if raw:
            try:
                cc = (_json.loads(raw) or {}).get(str(code))
            except Exception:
                cc = None
    if not cc:
        return []   # corp_code 매핑 없음 → DART 조회 불가(네이버 공시로 대체)

    try:
        bgn = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        resp = requests.get(
            "https://opendart.fss.or.kr/api/list.json",
            params={"crtfc_key": key, "corp_code": cc,
                    "bgn_de": bgn, "page_count": min(limit, 100)},
            headers={"User-Agent": _UA}, timeout=7,
        )
        data = resp.json()
        if data.get("status") != "000":
            return []
        out: list[dict] = []
        for d in (data.get("list") or [])[:limit]:
            nm = d.get("report_nm")
            if not nm:
                continue
            rno = d.get("rcept_no", "")
            dt = str(d.get("rcept_dt", ""))
            date = f"{dt[:4]}.{dt[4:6]}.{dt[6:8]}" if len(dt) == 8 else dt
            out.append({
                "title": nm,
                "link": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rno}" if rno else "",
                "date": date,
                "source_type": "dart",
            })
        return out
    except Exception:
        return []


def fetch_stock_history(code: str, market: str = "KOSPI") -> dict:
    """yfinance로 20일 종가 시계열 + 52주 고저 + 20일 평균 거래량.

    market: "KOSPI" or "KOSDAQ"
    """
    import yfinance as yf

    primary = ".KS" if market.upper() == "KOSPI" else ".KQ"
    fallback = ".KQ" if primary == ".KS" else ".KS"
    h = None
    symbol = f"{code}{primary}"
    for suffix in (primary, fallback):
        try:
            cand = yf.Ticker(f"{code}{suffix}").history(period="1y", auto_adjust=False)
        except Exception:
            cand = None
        if cand is not None and len(cand) >= 2:
            symbol = f"{code}{suffix}"
            h = cand
            break
    if h is None or len(h) < 2:
        return {}

    closes    = h["Close"].tolist()
    last20    = closes[-20:]
    high_52w  = float(h["High"].max())
    low_52w   = float(h["Low"].min())
    last_close = float(closes[-1])

    vols        = [float(v) for v in h["Volume"].tolist()[-20:] if v]
    vol_20d_avg = round(sum(vols) / len(vols)) if vols else None

    change_20d = None
    if len(last20) >= 2 and last20[0]:
        change_20d = round((last20[-1] - last20[0]) / last20[0] * 100, 2)

    pos_52w = None
    if high_52w > low_52w:
        pos_52w = round((last_close - low_52w) / (high_52w - low_52w) * 100, 1)

    return {
        "symbol":              symbol,
        "closes_20d":          [round(float(c), 2) for c in last20],
        "fifty_two_week_high": round(high_52w, 2),
        "fifty_two_week_low":  round(low_52w, 2),
        "last_close":          round(last_close, 2),
        "change_20d_pct":      change_20d,
        "pos_52w_pct":         pos_52w,
        "from_high_pct":       round((last_close - high_52w) / high_52w * 100, 2) if high_52w else None,
        "volume_20d_avg":      vol_20d_avg,
        "as_of":               str(h.index[-1].date()),
    }


def fetch_proxy_changes(tickers: list[str]) -> dict[str, float]:
    """미국 대리 티커의 전일 대비 등락률 (오버나이트 신호 계산용)."""
    import yfinance as yf

    out: dict[str, float] = {}
    for sym in tickers:
        try:
            h = yf.Ticker(sym).history(period="5d", auto_adjust=False)
            if len(h) < 2:
                continue
            prev = float(h.iloc[-2]["Close"])
            last = float(h.iloc[-1]["Close"])
            if prev:
                out[sym] = (last - prev) / prev * 100
        except Exception as e:
            print(f"[warn] proxy {sym}: {e}", file=sys.stderr)
    return out


def compute_overnight_signal(proxies: list[str], changes: dict[str, float]) -> dict:
    """프록시 티커 평균 등락 → direction 버킷 (up / neutral / down).

    임계값 ±0.7% 기준. 프록시 없으면 빈 dict 반환.
    """
    vals = [changes[p] for p in proxies if p in changes]
    if not vals:
        return {"direction": "", "avg_pct": None, "proxies": []}
    avg = sum(vals) / len(vals)
    if avg > 0.7:
        direction = "up"
    elif avg < -0.7:
        direction = "down"
    else:
        direction = "neutral"
    return {
        "direction": direction,
        "avg_pct":   round(avg, 2),
        "proxies":   [{"symbol": p, "change_pct": round(changes[p], 2)} for p in proxies if p in changes],
    }


# ── 통합 수집 ────────────────────────────────────────────────────────────────

def fetch_macro_context() -> dict:
    """거시 지표 전체를 병렬 수집 + 모듈 캐시 적용.

    [개선 내역]
      - 5개 독립 I/O를 ThreadPoolExecutor로 동시 실행 (순차→병렬)
      - 장중(3분) / 미국장(5분) / 장 외(10분) 가변 TTL 캐시
      - _MACRO_LOCK으로 동시 다중 요청 중 중복 수집 방지

    반환 구조:
      {
        "generated_at": str,
        "indices":    dict,   # KOSPI / KOSDAQ / KOSPI200
        "fx":         list,   # 환율
        "overnight":  list,   # 해외 시장
        "crypto_krw": dict,   # BTC / ETH
        "news":       list,   # 거시 뉴스
      }
    """
    # ── 캐시 확인 (락 없이 읽기 — 원자적 dict 접근으로 안전) ──────────────
    cached = _MACRO_CACHE["data"]
    if cached and (_time.monotonic() - _MACRO_CACHE["ts"]) < _macro_ttl_seconds():
        return cached

    # ── 캐시 미스: 락 획득 후 재확인 (Thundering-herd 방지) ───────────────
    with _MACRO_LOCK:
        cached = _MACRO_CACHE["data"]
        if cached and (_time.monotonic() - _MACRO_CACHE["ts"]) < _macro_ttl_seconds():
            return cached   # 다른 스레드가 이미 갱신한 경우

        result: dict = {"generated_at": datetime.now(KST).isoformat()}

        # 5개 독립 함수를 동시에 실행 (max_workers=5 → 각자 I/O 대기 중 병렬 진행)
        _TASKS: dict[str, tuple] = {
            "indices":    (fetch_market_indices,   {}),
            "fx":         (fetch_fx,               []),
            "overnight":  (fetch_overnight_markets, []),
            "crypto_krw": (fetch_upbit_crypto,     {}),
            "news":       (fetch_macro_news,       []),
        }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(fn): (key, default)
                for key, (fn, default) in _TASKS.items()
            }
            for future in as_completed(futures):
                key, default = futures[future]
                try:
                    result[key] = future.result()
                except Exception as e:
                    print(f"[warn] macro {key}: {e}", file=sys.stderr)
                    result[key] = default

        _MACRO_CACHE["data"] = result
        _MACRO_CACHE["ts"]   = _time.monotonic()

    return result


def fetch_stock_snapshot(
    code: str,
    market: str = "KOSPI",
    overnight_proxies: list[str] | None = None,
    proxy_changes: dict[str, float] | None = None,
) -> dict:
    """단일 종목 스냅샷 — 시세 + 뉴스 + 히스토리 + 오버나이트 신호.

    proxy_changes 를 미리 계산해서 넘기면 배치 수집 시 yfinance 중복 호출을 줄일 수 있음.
    """
    entry: dict = {"code": code, "market": market}
    if overnight_proxies and proxy_changes is not None:
        entry["overnight_signal"] = compute_overnight_signal(overnight_proxies, proxy_changes)
    elif overnight_proxies:
        changes = fetch_proxy_changes(overnight_proxies)
        entry["overnight_signal"] = compute_overnight_signal(overnight_proxies, changes)

    try:
        entry["quote"] = fetch_stock_quote(code)
    except Exception as e:
        print(f"[warn] quote {code}: {e}", file=sys.stderr)
        entry["quote"] = {}
    try:
        entry["history"] = fetch_stock_history(code, market)
    except Exception as e:
        print(f"[warn] history {code}: {e}", file=sys.stderr)
        entry["history"] = {}
    try:
        entry["news"] = fetch_stock_news(code)
    except Exception as e:
        print(f"[warn] news {code}: {e}", file=sys.stderr)
        entry["news"] = []
    try:
        entry["disclosures"] = fetch_stock_disclosures(code)
    except Exception as e:
        print(f"[warn] disclosures {code}: {e}", file=sys.stderr)
        entry["disclosures"] = []
    return entry


def fetch_stock_list_snapshot(
    stock_configs: list[dict],
) -> list[dict]:
    """여러 종목을 일괄 수집. stock_configs 각 항목 예:
      {"code":"005930","market":"KOSPI","overnight_proxy":["^SOX","NVDA"]}
    """
    all_proxies = sorted({
        p for s in stock_configs for p in (s.get("overnight_proxy") or [])
    })
    proxy_changes = fetch_proxy_changes(all_proxies) if all_proxies else {}

    result = []
    for cfg in stock_configs:
        snap = fetch_stock_snapshot(
            code               = cfg["code"],
            market             = cfg.get("market", "KOSPI"),
            overnight_proxies  = cfg.get("overnight_proxy"),
            proxy_changes      = proxy_changes,
        )
        # 이름/섹터/소유자 등 설정 필드 병합
        for key in ("name", "sector", "owners", "leader", "is_etf"):
            if cfg.get(key) is not None:
                snap[key] = cfg[key]
        result.append(snap)
    return result


def fetch_stock_list_quote_only(
    stock_configs: list[dict],
    *,
    max_workers: int = 10,
) -> list[dict]:
    """섹터 흐름용 경량 병렬 스냅샷.

    quote(시세)만 수집하고 history/news/disclosures는 건너뜀.
    ThreadPoolExecutor로 전 종목을 동시에 요청하므로
    fetch_stock_list_snapshot 대비 대폭 빠름.

    Args:
        stock_configs: [{"code":"005930","market":"KOSPI","sector":"반도체"}, ...]
        max_workers:   동시 스레드 수 (기본 10 — 네이버 서버 부하 고려)

    Returns:
        fetch_stock_list_snapshot 과 동일 구조이지만
        history/news/disclosures 키는 포함되지 않음.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(cfg: dict) -> dict:
        code   = cfg["code"]
        market = cfg.get("market", "KOSPI")
        snap: dict = {"code": code, "market": market}
        # 설정 필드(이름·섹터 등) 먼저 병합
        for key in ("name", "sector", "owners", "leader", "is_etf"):
            if cfg.get(key) is not None:
                snap[key] = cfg[key]
        try:
            snap["quote"] = fetch_stock_quote(code)
        except Exception as e:
            print(f"[warn] quote {code}: {e}", file=sys.stderr)
            snap["quote"] = {}
        return snap

    n = len(stock_configs)
    results: list[dict | None] = [None] * n
    workers = min(max_workers, n) if n else 1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_fetch_one, cfg): i
            for i, cfg in enumerate(stock_configs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                cfg = stock_configs[idx]
                print(f"[warn] sector quote failed {cfg.get('code')}: {e}", file=sys.stderr)
                results[idx] = {
                    "code":   cfg["code"],
                    "market": cfg.get("market", "KOSPI"),
                    "sector": cfg.get("sector"),
                    "quote":  {},
                }

    # None 보호 (이론상 발생 안 함)
    return [r for r in results if r is not None]


def fetch_stock_list_quote_cached(
    stock_configs: list[dict],
    *,
    max_workers: int = 15,
) -> list[dict]:
    """TTL 캐시 포함 섹터 흐름용 quote 수집.

    캐시 히트 시 즉시 반환 (0ms). 미스 시 fetch_stock_list_quote_only 실행 후 캐시 저장.
    Vercel 웜 인스턴스 재사용으로 반복 요청의 응답 속도를 대폭 단축.

    TTL: 장중(09:00~15:30 KST) 3분 / 장외·주말 10분
    """
    cached = _SECTOR_CACHE["data"]
    if cached and (_time.monotonic() - _SECTOR_CACHE["ts"]) < _sector_ttl_seconds():
        return cached

    with _SECTOR_LOCK:
        # Double-checked locking — 락 대기 중 다른 스레드가 갱신했을 수 있음
        cached = _SECTOR_CACHE["data"]
        if cached and (_time.monotonic() - _SECTOR_CACHE["ts"]) < _sector_ttl_seconds():
            return cached

        result = fetch_stock_list_quote_only(stock_configs, max_workers=max_workers)
        _SECTOR_CACHE["data"] = result
        _SECTOR_CACHE["ts"]   = _time.monotonic()

    return result
