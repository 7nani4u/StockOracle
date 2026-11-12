"""yfinance 쿠키 재사용 — 요청마다 쿠키를 다시 받아 모든 스레드가 한 줄로 서던 문제를 푼다 (2026-11-12 측정).

배경
  api/index.py 는 SQLite 캐시 오류를 피하려고 ``yf.cache.get_cookie_cache().dummy = True`` 로 yfinance 의 영속 쿠키 캐시를
  끈다. yfinance 1.2.0 의 ``YfData._get_cookie_basic`` 은 쿠키를 받은 직후에는 ``self._cookie`` 를 표시하지 않고, 다음 요청에서
  영속 캐시에서 읽어(``_load_cookie_curlCffi``) 표시한다. 캐시가 꺼져 있으면 그 표시가 영원히 생기지 않아 **모든 HTTP 요청이
  ``https://fc.yahoo.com`` 쿠키 요청을 먼저 보내고**, 그 요청은 전역 락(``_cookie_lock``) 안에서 일어난다. 그래서

    - 스레드를 늘려도 요청이 한 줄로 선다(처리량 = 1 / (쿠키 요청 시간 × 요청 수)),
    - ``Ticker.history`` 한 번이 HTTP 4회(쿠키 2 + 시간대 조회 1 + 시세 1), ``Ticker.info`` 한 번이 6회가 된다.

  이 PC 실측(docs/scan_engine_audit_20261112.md ⑧): 미국 122종목 ``yf.download`` 49.2초, ``Ticker.info`` 93종목 38.2초.

수정
  쿠키 요청이 성공해 세션 쿠키 항아리에 야후 쿠키가 들어 있으면 ``self._cookie = True`` 로 표시한다. 이는 csrf 전략
  (``_get_cookie_csrf``)이 이미 쓰는 표시이고, 영속 캐시가 켜진 정상 환경의 상태와 같다. 서버가 쿠키·크럼을 거부해 yfinance 가
  전략을 바꾸면(``_set_cookie_strategy``) 표시가 None 으로 돌아가 쿠키를 다시 받는다. 쿠키는 요청 인증일 뿐이라 받는 데이터는
  달라지지 않는다(종목·날짜별 비교는 같은 문서 참조).

복원: 환경변수 ``STOCKORACLE_YF_COOKIE_REUSE=0`` (이전 동작: 요청마다 쿠키를 다시 받음).
"""
from __future__ import annotations

import inspect
import os

_DISABLED = {"0", "false", "no", "off"}
_MARK = "_stockoracle_cookie_reuse"


def cookie_reuse_requested() -> bool:
    """환경변수 STOCKORACLE_YF_COOKIE_REUSE 가 끄지 않았는지(기본: 켬)."""
    return str(os.getenv("STOCKORACLE_YF_COOKIE_REUSE", "1")).strip().lower() not in _DISABLED


def _has_supported_signature(func) -> bool:
    """감쌀 메서드가 검증한 모양 ``(self, timeout=30)`` 인지(yfinance 1.2.0·1.8.0 소스 확인).

    ``requirements.txt`` 가 ``yfinance>=0.2.36`` 이라 구버전도 설치될 수 있다. 0.2.4x~0.2.5x 의 ``_get_cookie_basic`` 은
    ``(self, proxy=None, timeout=30)`` 이고 호출부가 위치 인자 두 개를 넘기며 ``_cookie`` 도 불리언 표시가 아니라 실제 쿠키
    값이다 — 그 버전에 이 감싸기를 씌우면 모든 요청이 ``TypeError`` 로 깨지거나 값의 뜻이 달라진다. 모양이 다르면 감싸지 않는다.
    """
    try:
        return list(inspect.signature(func).parameters) == ["self", "timeout"]
    except (TypeError, ValueError):
        return False


def _jar_has_yahoo_cookie(session) -> bool:
    """세션 쿠키 항아리에 야후 쿠키가 실제로 들어 있는지. 읽을 수 없으면 False(표시하지 않고 다음 요청에서 다시 받는다)."""
    try:
        return any("yahoo" in str(getattr(cookie, "domain", "")).lower() for cookie in session.cookies.jar)
    except Exception:
        return False


def enable_cookie_reuse(data_cls=None) -> bool:
    """``YfData._get_cookie_basic`` 이 받은 쿠키를 기억하게 한다. 켜졌으면 True, 꺼져 있거나 적용할 수 없으면 False.

    여러 번 불러도 한 번만 감싼다. 절대 raise 하지 않는다(적용 실패는 이전 동작으로 남는다).
    """
    try:
        if not cookie_reuse_requested():
            return False
        if data_cls is None:
            from yfinance.data import YfData as data_cls
        original = data_cls._get_cookie_basic
        if getattr(original, _MARK, False):
            return True
        if not _has_supported_signature(original):
            return False

        def _get_cookie_basic(self, *args, **kwargs):
            ok = original(self, *args, **kwargs)
            if ok and getattr(self, "_cookie", None) is None and _jar_has_yahoo_cookie(getattr(self, "_session", None)):
                self._cookie = True
            return ok

        setattr(_get_cookie_basic, _MARK, True)
        _get_cookie_basic.__wrapped__ = original
        data_cls._get_cookie_basic = _get_cookie_basic
        return True
    except Exception:
        return False


def disable_cookie_reuse(data_cls=None) -> bool:
    """감싼 것을 벗겨 원래 메서드로 되돌린다(테스트·진단용). 벗겼으면 True."""
    try:
        if data_cls is None:
            from yfinance.data import YfData as data_cls
        current = data_cls._get_cookie_basic
        if getattr(current, _MARK, False):
            data_cls._get_cookie_basic = current.__wrapped__
            return True
        return False
    except Exception:
        return False
