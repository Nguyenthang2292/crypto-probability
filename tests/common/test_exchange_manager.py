import pytest

from modules.ExchangeManager import ExchangeManager, PublicExchangeManager


class DummyExchange:
    def __init__(self):
        self.calls = 0

    def ping(self):
        self.calls += 1
        return "pong"


def test_public_manager_caches_instances(monkeypatch):
    public = PublicExchangeManager()

    class DummyCCXTModule:
        @staticmethod
        def dummy(params=None):
            return DummyExchange()

    monkeypatch.setattr("modules.ExchangeManager.ccxt", DummyCCXTModule)

    ex1 = public.connect_to_exchange_with_no_credentials("dummy")
    ex2 = public.connect_to_exchange_with_no_credentials("dummy")

    assert ex1 is ex2


def test_throttled_call_enforces_wait(monkeypatch):
    public = PublicExchangeManager(request_pause=0.01)
    exchange = DummyExchange()

    called = []

    def fake_time():
        return len(called) * 0.02

    monkeypatch.setattr("modules.ExchangeManager.time.time", fake_time)
    result = public.throttled_call(exchange.ping)
    called.append(1)
    result2 = public.throttled_call(exchange.ping)

    assert result == "pong" and result2 == "pong"
    assert exchange.calls == 2


def test_exchange_manager_normalizes_symbols():
    manager = ExchangeManager()
    assert manager.normalize_symbol("BTC/USDT:USDT") == "BTC/USDT"


def test_public_manager_rejects_unknown_exchange(monkeypatch):
    public = PublicExchangeManager()

    class DummyCCXTModule:
        pass

    monkeypatch.setattr("modules.ExchangeManager.ccxt", DummyCCXTModule)

    with pytest.raises(ValueError):
        public.connect_to_exchange_with_no_credentials("not_real")
