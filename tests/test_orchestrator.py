import pytest

import cpp_core
from python_slow_loop.orchestrator.arbiter import StrategyArbiter


@pytest.fixture
def arbiter():
    queue = cpp_core.SPSCQueue(10)
    return StrategyArbiter(queue)


def test_arbiter_normal_regime(arbiter):
    """Regime 0 (NORMAL): Trust the RL agent fully when uncertainty is low."""
    arbiter.current_regime = 0

    # RL requests 2.0 spread, +1.0 skew, with safely low uncertainty (0.5 sigma)
    success = arbiter.process_signals(
        spread_mu=2.0, spread_sigma=0.5, skew_mu=1.0, skew_sigma=0.5
    )
    assert success is True

    update = arbiter.queue.pop()
    assert update.spread_bps == 2.0
    assert update.skew_bps == 1.0
    assert update.agent_uncertainty == 0.5


def test_arbiter_volatility_regime(arbiter):
    """Regime 1 (VOLATILITY): Triple the spread, neutralize skew to 0.0."""
    arbiter.current_regime = 1

    # RL aggressively requests tight spread and heavy skew, but low uncertainty
    arbiter.process_signals(
        spread_mu=2.0, spread_sigma=0.5, skew_mu=5.0, skew_sigma=0.5
    )

    update = arbiter.queue.pop()
    assert update.spread_bps == 6.0  # 2.0 * 3.0 multiplier
    assert update.skew_bps == 0.0  # Skew clamped to neutral


def test_arbiter_bear_shock_regime(arbiter):
    """Regime 2 (BEAR_SHOCK): Widen spread, truncate any bullish skew."""
    arbiter.current_regime = 2

    # RL wants to quote bullishly (+3.0 skew) during a crash, low uncertainty
    arbiter.process_signals(
        spread_mu=2.0, spread_sigma=0.5, skew_mu=3.0, skew_sigma=0.5
    )

    update = arbiter.queue.pop()
    assert update.spread_bps == 3.0  # 2.0 * 1.5 multiplier
    assert update.skew_bps == 0.0  # Bullish skew intercepted and truncated to 0.0


def test_arbiter_epistemic_killswitch(arbiter):
    """OOD SHOCK: Veto AI entirely if ensemble variance exceeds safety limits."""
    arbiter.current_regime = 0  # Outwardly normal macroeconomic conditions

    # RL requests a standard 2.0 spread, but uncertainty is dangerously high (5.8 sigma > 3.0 limit)
    # This simulates the ensemble networks disagreeing violently due to novel market data
    arbiter.process_signals(
        spread_mu=2.0, spread_sigma=5.8, skew_mu=1.0, skew_sigma=2.0
    )

    update = arbiter.queue.pop()

    # Kill-switch should completely override the AI's requested 2.0 spread
    assert update.spread_bps == 15.0  # Max defensive spread activated
    assert update.skew_bps == 0.0  # Neutralized skew
    assert update.agent_uncertainty == 5.8  # Telemetry successfully passed down
