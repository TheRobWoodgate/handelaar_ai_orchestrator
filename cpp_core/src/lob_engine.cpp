#include "../include/lob_engine.h"
#include <iostream>
#include <iomanip>

LOBEngine::LOBEngine(SPSCQueue& queue, double max_pos, double max_spread)
    : queue_(queue), risk_guard_(max_pos, max_spread),
      active_spread_bps_(2.0), active_skew_bps_(0.0), active_regime_(0) {}

void LOBEngine::on_market_tick(double mid_price) {
    // 1. Poll the lock-free SPSC queue for updates from the Python AI Loop
    auto new_update = queue_.pop();
    if (new_update.has_value()) {
        active_spread_bps_ = new_update->spread_bps;
        active_skew_bps_ = new_update->skew_bps;
        active_regime_ = new_update->regime_id;
    }

    // 2. Mathematically map the relative spread and skew to absolute execution prices
    // Convert basis points to raw decimals (1 bps = 0.0001)
    double half_spread_price = mid_price * (active_spread_bps_ / 10000.0) / 2.0;
    double skew_price = mid_price * (active_skew_bps_ / 10000.0);

    double target_bid = mid_price - half_spread_price + skew_price;
    double target_ask = mid_price + half_spread_price + skew_price;

    // 3. Pass quotes through the Deterministic Pre-Trade Risk Guard
    bool spread_safe = risk_guard_.validate_spread(active_spread_bps_);
    bool bid_safe = risk_guard_.validate_quote(target_bid, 100.0, true);
    bool ask_safe = risk_guard_.validate_quote(target_ask, 100.0, false);

    // 4. Ultra-low-latency output logging (Simulating spdlog)
    if (spread_safe && bid_safe && ask_safe) {
        // In a real system, this logs to a lock-free ring buffer file stream
        std::cout << "[FAST LOOP] Mid: " << std::fixed << std::setprecision(2) << mid_price
                  << " | Quoting BID: " << target_bid << " ASK: " << target_ask
                  << " | Regime: " << active_regime_ << "\n";
    } else {
        std::cout << "[RISK ALERT] AI Proposed Parameters Breached Capital Guardrails. Orders Revoked.\n";
    }
}
