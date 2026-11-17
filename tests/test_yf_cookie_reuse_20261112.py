"""yfinance 쿠키 재사용(2026-11-12) 회귀 테스트 — 네트워크 없이.

앱은 SQLite 캐시 오류를 피하려고 yfinance 의 영속 쿠키 캐시를 끈다(api/index.py). 그러면 ``YfData._get_cookie_basic`` 이 받은 쿠키를
기억하지 못해 모든 HTTP 요청이 ``fc.yahoo.com`` 쿠키 요청을 먼저 보내고, 그 요청이 전역 락 안에서 일어나 스레드가 한 줄로 선다.
(이 PC 실측: 미국 122종목 yf.download 49.2초 → 8.0초, Ticker.info 93종목 38.2초 → 10.9초 — docs/scan_engine_audit_20261112.md ⑧)

여기서는 진짜 yfinance 코드(``YfData``)를 가짜 세션 위에서 돌려 (1) 결함 재현(요청마다 쿠키 재요청), (2) 수정 후 동시 호출에도 쿠키 요청이
한 번뿐, (3) 쿠키가 없으면 기억하지 않음, (4) 서버가 거부해 전략이 바뀌면 다시 받음, (5) 환경변수 스위치, (6) 라우트 응답의 추가 필드를
고정한다. yfinance 가 내부 구조를 바꾸면 (1)~(4)가 먼저 깨져 알려 준다.
"""

import threading
import time
from types import SimpleNamespace

import pytest
import yfinance
from yfinance.data import YfData

from market_briefing import yf_cookie


# ───────────────────────── 가짜 세션 ─────────────────────────

class _Jar(list):
    """CookieJar 처럼 순회하면 쿠키를 돌려주는 목록(yfinance 가 영속 캐시에 저장하려고 읽는 ``_cookies`` 는 비워 둔다)."""

    _cookies: dict = {}


class _Cookies:
    """curl_cffi ``Cookies`` 대역: ``jar`` 와 ``clear()``(전략을 바꿀 때 yfinance 가 부른다)."""

    def __init__(self):
        self.jar = _Jar()

    def clear(self):
        self.jar.clear()


class _FakeSession:
    """curl_cffi 세션 대역: fc.yahoo.com 요청은 쿠키를 항아리에 넣고, getcrumb 는 크럼 문자열을 준다."""

    def __init__(self, sets_cookie=True, latency=0.0):
        self.cookies = _Cookies()
        self.requests = []
        self.sets_cookie = sets_cookie
        self.latency = latency
        self._lock = threading.Lock()

    def get(self, url=None, **kwargs):
        if self.latency:
            time.sleep(self.latency)
        with self._lock:
            self.requests.append(url)
            if "fc.yahoo.com" in url and self.sets_cookie and not self.cookies.jar:
                self.cookies.jar.append(SimpleNamespace(domain=".yahoo.com", name="A3"))
        return SimpleNamespace(status_code=200, text="crumb-123", url=url, content=b"")

    def count(self, needle):
        return sum(1 for u in self.requests if needle in u)


class _NoPersistentCookieCache:
    """앱이 하는 것과 같다: 영속 쿠키 캐시를 끈다(lookup 은 항상 None, store 는 버린다)."""

    def lookup(self, strategy):
        return None

    def store(self, strategy, cookie):
        return None


@pytest.fixture
def app_like_yfinance(monkeypatch):
    """영속 쿠키 캐시가 꺼진 yfinance 와 (켜기 전의) 원래 ``_get_cookie_basic``.

    테스트가 끝나면 이 프로세스의 전역 상태(앱 import 가 이미 켠 쿠키 재사용)를 원래대로 돌려 다른 테스트에 번지지 않게 한다.
    """
    monkeypatch.setattr(yfinance.cache, "get_cookie_cache", lambda: _NoPersistentCookieCache())
    was_enabled = bool(getattr(YfData._get_cookie_basic, yf_cookie._MARK, False))
    yf_cookie.disable_cookie_reuse()
    monkeypatch.delenv("STOCKORACLE_YF_COOKIE_REUSE", raising=False)
    yield
    yf_cookie.disable_cookie_reuse()
    if was_enabled:
        yf_cookie.enable_cookie_reuse()


def _new_data(session):
    """싱글턴을 건드리지 않고 진짜 YfData 인스턴스를 만든다."""
    data = object.__new__(YfData)
    YfData.__init__(data)
    data._session = session
    return data


# ───────────────────────── (1) 결함 재현 ─────────────────────────

def test_without_cookie_reuse_every_request_fetches_the_cookie_again(app_like_yfinance):
    session = _FakeSession()
    data = _new_data(session)
    for _ in range(5):
        data._get_cookie_and_crumb()
    assert session.count("fc.yahoo.com") >= 5          # 호출마다 쿠키 재요청(첫 호출은 크럼을 받으며 한 번 더) — 락 안에서 직렬로 일어난다
    assert session.count("getcrumb") == 1               # 크럼은 처음 한 번만(self._crumb 가 유지됨)
    assert data._cookie is None


# ───────────────────────── (2) 수정 후 ─────────────────────────

def test_cookie_reuse_fetches_the_cookie_once_even_with_concurrent_callers(app_like_yfinance):
    assert yf_cookie.enable_cookie_reuse() is True
    session = _FakeSession(latency=0.002)
    data = _new_data(session)

    def worker():
        for _ in range(10):
            data._get_cookie_and_crumb()

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert session.count("fc.yahoo.com") == 1          # 120번 호출에 쿠키 요청은 단 한 번
    assert session.count("getcrumb") == 1
    assert data._cookie is True


def test_cookie_reuse_returns_the_same_crumb_and_strategy_as_before(app_like_yfinance):
    plain = _new_data(_FakeSession())
    crumb_before = plain._get_cookie_and_crumb()
    yf_cookie.enable_cookie_reuse()
    reused = _new_data(_FakeSession())
    assert reused._get_cookie_and_crumb() == crumb_before == ("crumb-123", "basic")
    assert reused._get_cookie_and_crumb() == crumb_before                       # 두 번째 호출도 같은 크럼·전략


# ───────────────────────── (3) 쿠키를 못 받았으면 기억하지 않는다 ─────────────────────────

def test_cookie_is_not_remembered_when_the_session_holds_no_yahoo_cookie(app_like_yfinance):
    yf_cookie.enable_cookie_reuse()
    session = _FakeSession(sets_cookie=False)
    data = _new_data(session)
    for _ in range(3):
        data._get_cookie_and_crumb()
    assert data._cookie is None
    assert session.count("fc.yahoo.com") >= 3          # 쿠키가 안 들어왔으니 호출마다 다시 시도(첫 호출은 크럼을 받으며 한 번 더)


def test_cookie_with_a_foreign_domain_is_not_taken_for_a_yahoo_cookie(app_like_yfinance):
    yf_cookie.enable_cookie_reuse()
    session = _FakeSession(sets_cookie=False)
    session.cookies.jar.append(SimpleNamespace(domain=".example.com", name="x"))
    data = _new_data(session)
    data._get_cookie_and_crumb()
    assert data._cookie is None


# ───────────────────────── (4) 서버가 거부해 전략이 바뀌면 다시 받는다 ─────────────────────────

def test_strategy_switch_forgets_the_remembered_cookie_and_fetches_again(app_like_yfinance):
    yf_cookie.enable_cookie_reuse()
    session = _FakeSession()
    data = _new_data(session)
    data._get_cookie_and_crumb()
    assert data._cookie is True and session.count("fc.yahoo.com") == 1

    data._set_cookie_strategy("csrf")                  # yfinance 가 401/403 을 받으면 하는 일(쿠키·크럼을 버린다)
    assert data._cookie is None and data._crumb is None
    data._set_cookie_strategy("basic")                 # csrf → basic 으로 돌아올 때 yfinance 가 쿠키 항아리를 비운다
    assert not session.cookies.jar
    data._get_cookie_and_crumb()
    assert session.count("fc.yahoo.com") == 2          # 새로 받았다
    assert data._cookie is True


# ───────────────────────── (5) 스위치·안전성 ─────────────────────────

class _Stub:
    """YfData 대역(플래그 동작만 본다)."""

    def __init__(self):
        self._cookie = None
        self._session = SimpleNamespace(cookies=SimpleNamespace(jar=_Jar([SimpleNamespace(domain=".yahoo.com")])))
        self.calls = 0

    def _get_cookie_basic(self, timeout=30):
        self.calls += 1
        return True


@pytest.mark.parametrize("value", ["0", "false", "No", "OFF", " off "])
def test_environment_switch_restores_the_previous_behavior(monkeypatch, value):
    monkeypatch.setenv("STOCKORACLE_YF_COOKIE_REUSE", value)
    class Local(_Stub):
        pass
    original = Local._get_cookie_basic
    assert yf_cookie.cookie_reuse_requested() is False
    assert yf_cookie.enable_cookie_reuse(Local) is False
    assert Local._get_cookie_basic is original          # 감싸지 않았다
    obj = Local()
    obj._get_cookie_basic()
    assert obj._cookie is None                          # 기억하지 않는다 = 이전 동작


def test_enable_wraps_once_and_disable_restores_the_original(monkeypatch):
    monkeypatch.delenv("STOCKORACLE_YF_COOKIE_REUSE", raising=False)
    class Local(_Stub):
        pass
    original = Local._get_cookie_basic
    assert yf_cookie.enable_cookie_reuse(Local) is True
    wrapped = Local._get_cookie_basic
    assert yf_cookie.enable_cookie_reuse(Local) is True
    assert Local._get_cookie_basic is wrapped and wrapped.__wrapped__ is original      # 두 번 불러도 한 겹
    obj = Local()
    assert obj._get_cookie_basic() is True and obj._cookie is True and obj.calls == 1
    assert yf_cookie.disable_cookie_reuse(Local) is True
    assert Local._get_cookie_basic is original
    assert yf_cookie.disable_cookie_reuse(Local) is False


class _OldYfinanceStub:
    """yfinance 0.2.4x~0.2.5x 모양: ``_get_cookie_basic(self, proxy=None, timeout=30)`` 이고 ``_cookie`` 는 실제 쿠키 값이다.

    ``requirements.txt`` 가 ``yfinance>=0.2.36`` 이라 이 버전도 설치될 수 있다(0.2.54 소스 확인).
    """

    def __init__(self):
        self._cookie = None
        self._session = SimpleNamespace(cookies=SimpleNamespace(jar=_Jar([SimpleNamespace(domain=".yahoo.com")])))

    def _get_cookie_basic(self, proxy=None, timeout=30):
        self._cookie = "A3=real-cookie-value"
        return self._cookie


def test_old_yfinance_signature_is_left_alone_instead_of_breaking_every_request(monkeypatch):
    monkeypatch.delenv("STOCKORACLE_YF_COOKIE_REUSE", raising=False)

    class Old(_OldYfinanceStub):
        pass
    original = Old._get_cookie_basic
    assert yf_cookie.enable_cookie_reuse(Old) is False              # 검증한 모양이 아니면 감싸지 않는다
    assert Old._get_cookie_basic is original
    obj = Old()
    assert obj._get_cookie_basic(None, 5) == "A3=real-cookie-value"     # 구버전 호출부(위치 인자 두 개)가 그대로 동작
    assert obj._cookie == "A3=real-cookie-value"                    # 실제 쿠키 값이 불리언 표시로 덮이지 않는다


def test_supported_shape_is_wrapped_and_arguments_pass_through_unchanged(monkeypatch):
    monkeypatch.delenv("STOCKORACLE_YF_COOKIE_REUSE", raising=False)
    seen = []

    class New(_Stub):
        def _get_cookie_basic(self, timeout=30):
            seen.append(timeout)
            return True

    assert yf_cookie._has_supported_signature(New._get_cookie_basic) is True
    assert yf_cookie.enable_cookie_reuse(New) is True
    obj = New()
    obj._get_cookie_basic()
    obj._get_cookie_basic(5)
    obj._get_cookie_basic(timeout=7)
    assert seen == [30, 5, 7]                                       # 인자를 바꾸거나 삼키지 않는다
    yf_cookie.disable_cookie_reuse(New)


def test_real_yfinance_matches_the_shape_the_wrapper_was_verified_against():
    # 이 PC 의 yfinance 가 검증한 모양이 아니면 쿠키 재사용이 조용히 꺼진다 — 그때 이 테스트가 먼저 알려 준다.
    assert yf_cookie._has_supported_signature(YfData._get_cookie_basic) is True


def test_enable_never_raises_on_an_unexpected_class():
    assert yf_cookie.enable_cookie_reuse(object) is False                              # 메서드가 없는 클래스
    assert yf_cookie.disable_cookie_reuse(object) is False


def test_unreadable_cookie_jar_means_no_memory_not_a_crash():
    class Broken(_Stub):
        pass
    Broken._get_cookie_basic = _Stub._get_cookie_basic
    yf_cookie.enable_cookie_reuse(Broken)
    obj = Broken()
    obj._session = SimpleNamespace(cookies=None)
    assert obj._get_cookie_basic() is True and obj._cookie is None
    yf_cookie.disable_cookie_reuse(Broken)


# ───────────────────────── (6) 앱이 켠 채로 뜬다 ─────────────────────────
# (스캔 응답의 ``data_quality.yf_cookie_reuse`` 는 tests/test_scan_pipeline_audit_20261112.py 가 고정한다.)

def test_app_starts_with_cookie_reuse_enabled_unless_the_switch_is_off():
    from api import index

    assert isinstance(index.YF_COOKIE_REUSE, bool)
    if yf_cookie.cookie_reuse_requested():
        assert index.YF_COOKIE_REUSE is True
        assert getattr(YfData._get_cookie_basic, yf_cookie._MARK, False) is True
