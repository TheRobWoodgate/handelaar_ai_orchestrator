import jax.numpy as jnp

from python_slow_loop.agents.rl_jax_agent import JAXRLAgent


def test_agent_initialization():
    """Ensure functional purity and stateless parameter initialization."""
    agent = JAXRLAgent(state_dim=10, action_dim=2, seed=42)
    assert agent.params["w1"].shape == (10, 64)
    assert agent.params["w2"].shape == (64, 2)


def test_agent_output_bounds():
    """Verify that the XLA-compiled forward pass maps outputs safely via tanh."""
    agent = JAXRLAgent(state_dim=10, action_dim=2)

    # Test with a zero state
    state = jnp.zeros(10)
    spread, skew = agent.select_action(state)

    # Based on our mapping logic in rl_jax_agent.py:
    # Spread should be mapped between 1.0 and 15.0 bps
    # Skew should be mapped between -5.0 and 5.0 bps
    assert 1.0 <= spread <= 15.0
    assert -5.0 <= skew <= 5.0

    # Test with extreme state (simulating a market shock)
    extreme_state = jnp.ones(10) * 100.0
    spread_ext, skew_ext = agent.select_action(extreme_state)

    assert 1.0 <= spread_ext <= 15.0
    assert -5.0 <= skew_ext <= 5.0
