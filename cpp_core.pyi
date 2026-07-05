# cpp_core.pyi
# Static type stubs for the compiled pybind11 C++ module

class StrategyUpdate:
    spread_bps: float
    skew_bps: float
    regime_id: int
    agent_uncertainty: float
    def __init__(
        self,
        spread_bps: float,
        skew_bps: float,
        regime_id: int,
        agent_uncertainty: float,
    ) -> None: ...

class SPSCQueue:
    def __init__(self, capacity: int) -> None: ...
    def push(self, update: StrategyUpdate) -> bool: ...
    def pop(self) -> StrategyUpdate | None: ...

class LOBEngine:
    def __init__(self, queue: SPSCQueue, max_pos: float, max_spread: float) -> None: ...
    # NEW: The thread pinning method
    def pin_thread_to_core(self, core_id: int) -> None: ...
    # UPDATED: The 4-variable L1 tick engine
    def on_market_tick(
        self, bid_price: float, bid_qty: float, ask_price: float, ask_qty: float
    ) -> None: ...
