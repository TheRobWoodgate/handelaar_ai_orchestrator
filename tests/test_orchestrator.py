import pytest

import cpp_core
from python_slow_loop.orchestrator.arbiter import StrategyArbiter


@pytest.fixture
def arbiter():
    queue = cpp_core.SPSCQueue(10)
    return StrategyArbiter(queue)


def test_arbiter_normal_regime(arbiter):
    """Regime 0 (NORMAL): Trust the RL agent fully."""
    arbiter.current_regime = 0
    # RL requests 2.0 spread, +1.0 skew
    success = arbiter.process_signals(2.0, 1.0)
    assert success is True

    update = arbiter.queue.pop()
    assert update.spread_bps == 2.0
    assert update.skew_bps == 1.0


def test_arbiter_volatility_regime(arbiter):
    """Regime 1 (VOLATILITY): Triple the spread, neutralize skew to 0.0."""
    arbiter.current_regime = 1
    # RL aggressively requests tight spread and heavy skew
    arbiter.process_signals(2.0, 5.0)

    update = arbiter.queue.pop()
    assert update.spread_bps == 6.0  # 2.0 * 3.0 multiplier
    assert update.skew_bps == 0.0  # Skew clamped to neutral


def test_arbiter_bear_shock_regime(arbiter):
    """Regime 2 (BEAR_SHOCK): Widen spread, truncate any bullish skew."""
    arbiter.current_regime = 2

    # RL hallucinated and wants to quote bullishly (+3.0 skew) during a crash
    arbiter.process_signals(2.0, 3.0)

    update = arbiter.queue.pop()
    assert update.spread_bps == 3.0  # 2.0 * 1.5 multiplier
    assert update.skew_bps == 0.0  # Bullish skew intercepted and truncated to 0.0
