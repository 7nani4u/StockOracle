"""데이터 수집 레이어 — Naver Finance 스크래핑 + yfinance + Upbit API.

출처: k-ant-daily/scripts/fetch_news.py 에서 핵심 함수 이식.
StockOracle용으로 독립 함수 형태로 재구성 (파일 I/O 없음, 순수 반환값).

주요 함수:
  fetch_macro_context()       — 거시 지표 통합 수집 (지수 + FX + 간밤 + 뉴스)
  fetch_stock_snapshot()      — 단일 종목 스냅샷 (시세 + 뉴스 + 히스토리 + 오버나이트)
  fetch_stock_list_snapshot() — 종목 리스트 일괄 수집
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup

KST = timezone(timedelta(hours=9))

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


# ── 내부 HTTP 헬퍼 ──────────────────────────────────────────────────────────

def _get(url: str, referer: str | None = None) -> BeautifulSoup:
    headers = {"User-Agent": _UA, "Accept-Language": "ko-KR,ko;q=0.9"}
    if referer:
        headers["Referer"] = referer
    r = requests.get(url, headers=headers, timeout=15)
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
    """간밤 해외 시장 (S&P500, 나스닥, VIX, WTI, 금, BTC, 달러인덱스) via yfinance."""
    import yfinance as yf

    tickers = [
        ("^GSPC",     "S&P 500"),
        ("^DJI",      "다우존스"),
        ("^IXIC",     "나스닥"),
        ("^VIX",      "VIX"),
        ("^KS200",    "KOSPI200 (종가)"),
        ("CL=F",      "WTI 원유"),
        ("GC=F",      "금"),
        ("BTC-USD",   "비트코인"),
        ("DX-Y.NYB",  "달러인덱스"),
    ]
    out: list[dict] = []
    for sym, label in tickers:
        try:
            h = yf.Ticker(sym).history(period="5d", auto_adjust=False)
            if len(h) < 2:
                continue
            prev = float(h.iloc[-2]["Close"])
            last = float(h.iloc[-1]["Close"])
            diff = last - prev
            pct  = (diff / prev * 100) if prev else 0.0
            pct_sign = "+" if pct >= 0 else ""
            abs_sign = "+" if diff >= 0 else "-"
            out.append({
                "symbol":     sym,
                "name":       label,
                "value":      f"{last:,.2f}",
                "change":     f"{pct_sign}{pct:.2f}%",
                "change_abs": f"{abs_sign}{abs(diff):,.2f}",
                "change_pct": round(pct, 2),
                "direction":  "up" if pct > 0 else ("down" if pct < 0 else "flat"),
                "as_of":      str(h.index[-1].date()),
            })
        except Exception as e:
            print(f"[warn] yfinance {sym}: {e}", file=sys.stderr)
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

def fetch_stock_quote(code: str) -> dict:
    """네이버 금융에서 현재 시세 (가격, 등락, 거래량)."""
    url  = f"https://finance.naver.com/item/main.naver?code={code}"
    soup = _get(url)
    out: dict = {}
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
    """거시 지표 전체를 한 번에 수집해서 반환.

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
    result: dict = {
        "generated_at": datetime.now(KST).isoformat(),
    }
    try:
        result["indices"] = fetch_market_indices()
    except Exception as e:
        print(f"[warn] indices: {e}", file=sys.stderr)
        result["indices"] = {}
    try:
        result["fx"] = fetch_fx()
    except Exception as e:
        print(f"[warn] fx: {e}", file=sys.stderr)
        result["fx"] = []
    try:
        result["overnight"] = fetch_overnight_markets()
    except Exception as e:
        print(f"[warn] overnight: {e}", file=sys.stderr)
        result["overnight"] = []
    try:
        result["crypto_krw"] = fetch_upbit_crypto()
    except Exception as e:
        print(f"[warn] crypto: {e}", file=sys.stderr)
        result["crypto_krw"] = {}
    try:
        result["news"] = fetch_macro_news()
    except Exception as e:
        print(f"[warn] macro news: {e}", file=sys.stderr)
        result["news"] = []
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
