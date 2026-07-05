#pragma once
#include "spsc_queue.h"
#include "risk_guard.h"
#include <string>

class LOBEngine {
private:
    SPSCQueue& queue_;
    RiskGuard risk_guard_;

    // Active AI parameters
    double active_spread_bps_;
    double active_skew_bps_;
    int active_regime_;
    double active_uncertainty_;

    // Microstructure State Tracking (for OFI)
    double prev_bid_price_;
    double prev_bid_qty_;
    double prev_ask_price_;
    double prev_ask_qty_;

public:
    LOBEngine(SPSCQueue& queue, double max_pos, double max_spread);

    // OS-Level Hardware Sympathy
    void pin_thread_to_core(int core_id);

    // Refactored to ingest L1 Top-of-Book data instead of just a mid price
    void on_market_tick(double bid_price, double bid_qty, double ask_price, double ask_qty);
};
