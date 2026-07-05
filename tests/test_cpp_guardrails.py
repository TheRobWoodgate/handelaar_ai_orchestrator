import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

import cpp_core
from python_slow_loop.orchestrator.arbiter import AgentUpdate


def test_spsc_queue_capacity():
    """Prove the lock-free queue rejects overflows gracefully rather than segfaulting."""
    queue = cpp_core.SPSCQueue(3)
    # UPDATED: Added the 4th argument (agent_uncertainty = 0.5)
    update = cpp_core.StrategyUpdate(2.0, 0.0, 0, 0.5)

    assert queue.push(update) is True
    assert queue.push(update) is True
    assert queue.push(update) is False  # Queue is full, must return False


@given(
    spread=st.floats(min_value=-1000.0, max_value=1000.0),
    skew=st.floats(min_value=-1000.0, max_value=1000.0),
    regime=st.integers(min_value=-5, max_value=10),
    uncertainty=st.floats(min_value=-10.0, max_value=10.0),  # NEW
)
def test_pydantic_boundary_guardrails(spread, skew, regime, uncertainty):
    """Property-based testing: Ensure structural validation intercepts malformed AI data."""
    is_valid_spread = 1.0 <= spread <= 50.0
    is_valid_skew = -20.0 <= skew <= 20.0
    is_valid_regime = 0 <= regime <= 3
    is_valid_uncertainty = uncertainty >= 0.0  # NEW: Variance cannot be negative

    if not (
        is_valid_spread and is_valid_skew and is_valid_regime and is_valid_uncertainty
    ):
        # We expect a validation error if the AI hallucinates outside physical bounds
        with pytest.raises(ValidationError):
            AgentUpdate(
                spread_bps=spread,
                skew_bps=skew,
                regime_id=regime,
                agent_uncertainty=uncertainty,
            )
