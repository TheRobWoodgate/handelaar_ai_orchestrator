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

    @field_validator("spread_bps")
    def clamp_aggressive_spreads(cls, v):
        """Mathematically prevent the AI from quoting unsafely tight spreads."""
        if v < 1.0:
            logging.warning("Agent attempted spread < 1.0 bps. Hard clamping to 1.0.")
            return 1.0
        return v


# --- The Arbiter ---
class StrategyArbiter:
    __slots__ = ("queue", "current_regime", "risk_matrix")

    def __init__(self, spsc_queue: cpp_core.SPSCQueue):
        self.queue = spsc_queue
        self.current_regime = 0  # Default to NORMAL state

        # Formal Logic Decision Matrix: dict mapping Regime ID -> (Spread_Multiplier, Skew_Clamp)
        # This resolves conflicts between the RL agent and the LLM.
        self.risk_matrix = {
            0: (1.0, 1.0),  # NORMAL: Trust the RL agent fully.
            1: (
                3.0,
                0.0,
            ),  # VOLATILITY: Triple the spread, neutralize skew to 0.0 to avoid directional risk.
            2: (
                1.5,
                -1.0,
            ),  # BEAR_SHOCK: Widen spread, force skew to be strictly negative (bid-heavy).
            3: (
                1.5,
                1.0,
            ),  # BULL_SHOCK: Widen spread, force skew to be strictly positive (ask-heavy).
        }

    def update_macro_regime(self, llm_signal: dict):
        """Asynchronously updates the risk state without triggering an execution push."""
        if llm_signal and llm_signal.get("confidence", 0.0) >= 0.75:
            new_regime = llm_signal.get("regime_id", 0)
            if self.current_regime != new_regime:
                logging.info(
                    f"Macro Regime Shift Detected: Transacting from {self.current_regime} to {new_regime}"
                )
                self.current_regime = new_regime

    def process_signals(
        self, rl_spread: float, rl_skew: float, llm_signal: dict | None = None
    ) -> bool:
        """
        Weights competing probabilistic signals, enforces structural boundaries,
        and pushes the final parameters into the C++ Fast Loop memory block.
        """
        # 1. Evaluate LLM Signal
        if llm_signal and llm_signal.get("confidence", 0.0) > 0.85:
            if self.current_regime != llm_signal["regime_id"]:
                logging.info(
                    f"Macro Regime Shift Detected: Transacting to {llm_signal['regime_id']}"
                )
                self.current_regime = llm_signal["regime_id"]

        # 2. Apply Formal Logic Risk Matrix
        spread_mult, skew_clamp = self.risk_matrix.get(self.current_regime, (1.0, 1.0))
        adjusted_spread = rl_spread * spread_mult

        # Directional Skew Neutralization
        if skew_clamp == 0.0:
            adjusted_skew = 0.0
        elif skew_clamp < 0.0:
            adjusted_skew = min(
                rl_skew, 0.0
            )  # Truncate any bullish RL skew during a bear shock
        else:
            adjusted_skew = max(
                rl_skew, 0.0
            )  # Truncate any bearish RL skew during a bull shock

        # 3. Deterministic Data Contract Validation
        try:
            validated = AgentUpdate(
                spread_bps=adjusted_spread,
                skew_bps=adjusted_skew,
                regime_id=self.current_regime,
            )
        except Exception as e:
            logging.error(
                f"Runtime validation failed. AI Output Malformed: {e}. Dropping update."
            )
            return False

        # 4. Zero-Copy Push to C++
        cpp_update = cpp_core.StrategyUpdate(
            validated.spread_bps, validated.skew_bps, validated.regime_id
        )

        success = self.queue.push(cpp_update)
        if not success:
            # SPSC push returns False if the queue is full (Python is producing faster than C++ can consume)
            logging.debug("SPSC Queue full. Dropping update to prevent blocking.")

        return success
