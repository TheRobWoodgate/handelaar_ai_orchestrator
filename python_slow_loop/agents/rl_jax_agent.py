import jax
import jax.numpy as jnp


class JAXRLAgent:
    def __init__(self, state_dim: int, action_dim: int, seed: int = 42):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize pseudo-random number generator key for JAX
        self.rng_key = jax.random.PRNGKey(seed)

        # Initialize network parameters (weights & biases)
        self.params = self._init_network_params()

    def _init_network_params(self):
        """Functionally pure initialization of layer weights."""
        k1, k2 = jax.random.split(self.rng_key)

        # Weights for a simple hidden layer (64 neurons) and output layer
        w1 = jax.random.normal(k1, (self.state_dim, 64)) * jnp.sqrt(
            2.0 / self.state_dim
        )
        b1 = jnp.zeros((64,))

        w2 = jax.random.normal(k2, (64, self.action_dim)) * jnp.sqrt(2.0 / 64)
        b2 = jnp.zeros((self.action_dim,))

        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    @staticmethod
    @jax.jit
    def forward(params, state):
        """
        Hyper-optimized forward pass using XLA Compilation.
        Takes state (LOB metrics) and returns continuous actions (Spread, Skew).
        """
        # Hidden layer with ReLU activation
        hidden = jnp.dot(state, params["w1"]) + params["b1"]
        hidden = jax.nn.relu(hidden)

        # Output layer
        output = jnp.dot(hidden, params["w2"]) + params["b2"]

        # Tanh activation bounds our actions between -1 and 1
        scaled_output = jnp.tanh(output)

        # Action mapping:
        # Action 0 -> Spread (mapped to 1.0 to 15.0 bps)
        # Action 1 -> Skew   (mapped to -5.0 to 5.0 bps)
        spread = 1.0 + (scaled_output[0] + 1.0) * 7.0
        skew = scaled_output[1] * 5.0

        return jnp.array([spread, skew])

    def select_action(self, state_array):
        """Thread-safe interface for the orchestrator loop."""
        state_jnp = jnp.array(state_array, dtype=jnp.float32)
        # Execute the JIT-compiled forward pass
        actions = self.forward(self.params, state_jnp)
        return float(actions[0]), float(actions[1])
