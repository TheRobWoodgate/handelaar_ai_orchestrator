import logging

from pydantic import BaseModel, Field, field_validator

import cpp_core  # Our pybind11 module


# --- Data Contracts ---
class AgentUpdate(BaseModel):
    """Runtime structural validation to guarantee downstream integrity."""

    spread_bps: float = Field(
        ..., ge=1.0, le=50.0, description="Spread in basis points"
    )
    skew_bps: float = Field(..., ge=-20.0, le=20.0, description="Skew in basis points")
    regime_id: int = Field(
        ..., ge=0, le=3, description="0:Normal, 1:Vol, 2:Bear, 3:Bull"
    )
    agent_uncertainty: float = Field(..., ge=0.0)

    @field_validator("spread_bps")
    def clamp_aggressive_spreads(cls, v):
        """Mathematically prevent the AI from quoting unsafely tight spreads."""
        if v < 1.0:
            logging.warning("Agent attempted spread < 1.0 bps. Hard clamping to 1.0.")
            return 1.0
        return v


# --- The Arbiter ---
class StrategyArbiter:
    def __init__(self, spsc_queue: cpp_core.SPSCQueue):
        self.queue = spsc_queue
        self.current_regime = 0  # Default: Normal Market

        # Risk Limits
        self.MAX_EPISTEMIC_UNCERTAINTY = 3.0  # Volatility bps variance threshold

    def update_macro_regime(self, llm_signal: dict | None = None):
        """Asynchronously updates the risk state from background LLM macro analysis."""
        if llm_signal and llm_signal.get("confidence", 0.0) >= 0.75:
            new_regime = llm_signal.get("regime_id", 0)
            if self.current_regime != new_regime:
                logging.info(
                    f"Macro Regime Shift Detected: Transacting from {self.current_regime} to {new_regime}"
                )
                self.current_regime = new_regime

    def process_signals(
        self, spread_mu: float, spread_sigma: float, skew_mu: float, skew_sigma: float
    ) -> bool:
        """
        Ingests continuous probabilistic signals from the JAX Deep Ensemble,
        applies formal risk overrides, and flashes validated states down to the C++ metal.
        """
        final_spread = spread_mu
        final_skew = skew_mu

        # ---------------------------------------------------------------------
        # CRITICAL SAFETY LAYER 1: Epistemic Uncertainty (OOD Kill-Switch)
        # ---------------------------------------------------------------------
        if spread_sigma > self.MAX_EPISTEMIC_UNCERTAINTY:
            logging.warning(
                f"[PROBABILISTIC VETO] AI uncertainty spiked (σ = {spread_sigma:.2f} bps). "
                f"Out-of-Distribution state detected. Activating Kill-Switch."
            )
            # Force absolute defensive posture
            final_spread = 15.0  # Max wide spread to guarantee safety
            final_skew = 0.0  # Completely neutralize inventory bias

            # Form the structural packet and bypass downstream macro multipliers
            try:
                packet = AgentUpdate(
                    spread_bps=final_spread,
                    skew_bps=final_skew,
                    regime_id=self.current_regime,
                    agent_uncertainty=spread_sigma,
                )
                update_raw = cpp_core.StrategyUpdate(
                    packet.spread_bps,
                    packet.skew_bps,
                    packet.regime_id,
                    packet.agent_uncertainty,
                )
                return self.queue.push(update_raw)
            except Exception as e:
                logging.error(f"Kill-switch structural generation failed: {e}")
                return False

        # ---------------------------------------------------------------------
        # LAYER 2: Exogenous Macro-Regime Overrides (Deterministic Logic Matrix)
        # ---------------------------------------------------------------------
        if self.current_regime == 1:  # VOLATILITY
            final_spread = spread_mu * 3.0
            final_skew = 0.0  # Flatten directional risk
        elif self.current_regime == 2:  # BEAR_SHOCK
            final_spread = spread_mu * 1.5
            if skew_mu > 0.0:
                final_skew = 0.0  # Forcefully truncate bullish bias during market crash

        # ---------------------------------------------------------------------
        # LAYER 3: Boundary Guardrail Serialization & Push
        # ---------------------------------------------------------------------
        try:
            # Enforce structural contracts via Pydantic
            validated_update = AgentUpdate(
                spread_bps=final_spread,
                skew_bps=final_skew,
                regime_id=self.current_regime,
                agent_uncertainty=spread_sigma,
            )

            # Serialize directly into the zero-copy C++ queue
            native_update = cpp_core.StrategyUpdate(
                validated_update.spread_bps,
                validated_update.skew_bps,
                validated_update.regime_id,
                validated_update.agent_uncertainty,
            )
            return self.queue.push(native_update)

        except Exception as e:
            logging.error(
                f"Runtime validation failed. AI Output Malformed: {e}. Dropping update."
            )
            return False
