#pragma once
#include "spsc_queue.h"
#include "risk_guard.h"
#include <string>

class LOBEngine {
private:
    SPSCQueue& queue_;
    RiskGuard risk_guard_;

    // Active parameters modified asynchronously by the AI orchestrator
    double active_spread_bps_;
    double active_skew_bps_;
    int active_regime_;

public:
    LOBEngine(SPSCQueue& queue, double max_pos, double max_spread);

    // The core execution loop function triggered on every microsecond market tick
    void on_market_tick(double mid_price);
};
