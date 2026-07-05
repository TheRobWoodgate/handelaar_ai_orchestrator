import jax
import jax.numpy as jnp


class ProbabilisticJAXAgent:
    def __init__(
        self, state_dim: int, action_dim: int, ensemble_size: int = 5, seed: int = 42
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = ensemble_size  # K = Number of independent networks in the ensemble

        self.rng_key = jax.random.PRNGKey(seed)

        # Initialize K distinct sets of weights as a single batched tensor
        self.params = self._init_ensemble_params()

        # CRITICAL: XLA-Compile the vectorized forward pass.
        # in_axes=(0, None) tells JAX: "Batch across the parameters (axis 0),
        # but broadcast the exact same LOB state (None) to every network."
        self.forward_vmap = jax.jit(jax.vmap(self._forward_single, in_axes=(0, None)))

    def _init_ensemble_params(self):
        """Initializes K distinct neural networks with different random seeds."""
        # Generate K unique PRNG subkeys
        keys = jax.random.split(self.rng_key, self.k)

        def init_single(key):
            """Initializes a single network."""
            k1, k2 = jax.random.split(key)
            w1 = jax.random.normal(k1, (self.state_dim, 64)) * jnp.sqrt(
                2.0 / self.state_dim
            )
            b1 = jnp.zeros((64,))
            w2 = jax.random.normal(k2, (64, self.action_dim)) * jnp.sqrt(2.0 / 64)
            b2 = jnp.zeros((self.action_dim,))
            return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

        # Vectorize the initialization to return a single dictionary of stacked tensors
        return jax.vmap(init_single)(keys)

    @staticmethod
    def _forward_single(params, state):
        """The deterministic forward pass for a SINGLE network."""
        hidden = jnp.dot(state, params["w1"]) + params["b1"]
        hidden = jax.nn.relu(hidden)
        output = jnp.dot(hidden, params["w2"]) + params["b2"]

        scaled_output = jnp.tanh(output)
        spread = 1.0 + (scaled_output[0] + 1.0) * 7.0
        skew = scaled_output[1] * 5.0
        return jnp.array([spread, skew])

    def select_action(self, state_array):
        """Executes the ensemble and calculates Epistemic Uncertainty (Variance)."""
        state_jnp = jnp.array(state_array, dtype=jnp.float32)

        # Execute all 5 networks in parallel on the GPU
        # Returns shape: (K, 2) -> 5 rows of [spread, skew]
        ensemble_outputs = self.forward_vmap(self.params, state_jnp)

        # Calculate the mathematical mean (mu) and standard deviation (sigma) across the K networks
        mu = jnp.mean(ensemble_outputs, axis=0)
        sigma = jnp.std(ensemble_outputs, axis=0)

        # Return the probability distributions: Spread (μ, σ) and Skew (μ, σ)
        return float(mu[0]), float(sigma[0]), float(mu[1]), float(sigma[1])
