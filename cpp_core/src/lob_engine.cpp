#include "../include/lob_engine.h"
#include <iostream>
#include <iomanip>
#include <pthread.h>

// Platform-specific headers for thread affinity
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/thread_policy.h>
#endif

LOBEngine::LOBEngine(SPSCQueue& queue, double max_pos, double max_spread)
    : queue_(queue), risk_guard_(max_pos, max_spread),
      active_spread_bps_(2.0), active_skew_bps_(0.0), active_regime_(0), active_uncertainty_(0.0),
      prev_bid_price_(0.0), prev_bid_qty_(0.0), prev_ask_price_(0.0), prev_ask_qty_(0.0) {}


void LOBEngine::pin_thread_to_core(int core_id) {
    // Locks the current execution thread to a specific physical CPU core
#ifdef __APPLE__
    // macOS Apple Silicon implementation
    thread_affinity_policy_data_t policy = { core_id };
    thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_AFFINITY_POLICY,
                      (thread_policy_t)&policy, 1);
    std::cout << "[SYSTEM] C++ Execution Loop pinned to Apple Silicon Affinity Tag: " << core_id << "\n";
#else
    // Standard Linux production implementation
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    std::cout << "[SYSTEM] C++ Execution Loop pinned to Linux CPU Core: " << core_id << "\n";
#endif
}


void LOBEngine::on_market_tick(double bid_price, double bid_qty, double ask_price, double ask_qty) {
    // 1. Calculate Order Flow Imbalance (OFI)
    double bid_power = 0.0;
    if (bid_price > prev_bid_price_) bid_power = bid_qty;
    else if (bid_price == prev_bid_price_) bid_power = bid_qty - prev_bid_qty_;
    else bid_power = -prev_bid_qty_;

    double ask_power = 0.0;
    if (ask_price < prev_ask_price_) ask_power = ask_qty;
    else if (ask_price == prev_ask_price_) ask_power = ask_qty - prev_ask_qty_;
    else ask_power = -prev_ask_qty_;

    [[maybe_unused]] double current_ofi = bid_power - ask_power;

    // Update state for the next tick
    prev_bid_price_ = bid_price;
    prev_bid_qty_ = bid_qty;
    prev_ask_price_ = ask_price;
    prev_ask_qty_ = ask_qty;

    // Calculate mid price from L1 data
    double mid_price = (bid_price + ask_price) / 2.0;

    // 2. Poll the lock-free SPSC queue
    auto new_update = queue_.pop();
    if (new_update.has_value()) {
        active_spread_bps_ = new_update->spread_bps;
        active_skew_bps_ = new_update->skew_bps;
        active_regime_ = new_update->regime_id;
        active_uncertainty_ = new_update->agent_uncertainty;
    }

    // 3. Mathematically map the relative spread and skew to absolute execution prices
    double half_spread_price = mid_price * (active_spread_bps_ / 10000.0) / 2.0;
    double skew_price = mid_price * (active_skew_bps_ / 10000.0);

    double target_bid = mid_price - half_spread_price + skew_price;
    double target_ask = mid_price + half_spread_price + skew_price;

    // 4. Pass quotes through the Deterministic Pre-Trade Risk Guard
    bool spread_safe = risk_guard_.validate_spread(active_spread_bps_);
    bool bid_safe = risk_guard_.validate_quote(target_bid, 100.0, true);
    bool ask_safe = risk_guard_.validate_quote(target_ask, 100.0, false);

    // 5. Output logging (Now includes OFI)
    if (spread_safe && bid_safe && ask_safe) {
        // Suppress printing on every single micro-tick in production,
        // but for our test harness we want to see the OFI calculations
        // std::cout << "[FAST LOOP] Mid: " << std::fixed << std::setprecision(2) << mid_price
        //          << " | OFI: " << current_ofi << "\n";
    }
}
