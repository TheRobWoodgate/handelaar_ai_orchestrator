import jax.numpy as jnp

# UPDATED: Import the new agent class
from python_slow_loop.agents.rl_jax_agent import ProbabilisticJAXAgent


def test_agent_initialization():
    """Ensure functional purity and stateless parameter initialization across the ensemble."""
    agent = ProbabilisticJAXAgent(state_dim=10, action_dim=2, ensemble_size=5, seed=42)

    # UPDATED: The params are now batched across K=5 networks via jax.vmap
    # Shape is now (Batch, In, Out)
    assert agent.params["w1"].shape == (5, 10, 64)
    assert agent.params["w2"].shape == (5, 64, 2)


def test_agent_output_bounds():
    """Verify that the XLA-compiled forward pass maps outputs safely and calculates variance."""
    agent = ProbabilisticJAXAgent(state_dim=10, action_dim=2, ensemble_size=5)

    # Test with a zero state
    state = jnp.zeros(10)

    # UPDATED: Unpack all 4 probabilistic variables
    spread_mu, spread_sigma, skew_mu, skew_sigma = agent.select_action(state)

    assert 1.0 <= spread_mu <= 15.0
    assert -5.0 <= skew_mu <= 5.0
    assert spread_sigma >= 0.0  # Standard deviation must be mathematically positive
    assert skew_sigma >= 0.0

    # Test with extreme state (simulating a market shock)
    extreme_state = jnp.ones(10) * 100.0
    spread_mu_ext, spread_sigma_ext, skew_mu_ext, skew_sigma_ext = agent.select_action(
        extreme_state
    )

    assert 1.0 <= spread_mu_ext <= 15.0
    assert -5.0 <= skew_mu_ext <= 5.0
    assert spread_sigma_ext >= 0.0
    assert skew_sigma_ext >= 0.0
