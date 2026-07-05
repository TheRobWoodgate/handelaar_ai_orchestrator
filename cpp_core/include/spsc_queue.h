#pragma once
#include <atomic>
#include <vector>
#include <cstddef>
#include <optional>

// Align to 64 bytes to prevent false sharing between CPU cache lines.
// This represents the combined output of our RL Agent (Spread/Skew) and LLM (Regime).
struct StrategyUpdate {
    double spread_bps;
    double skew_bps;
    int regime_id;
    double agent_uncertainty; // NEW: Telemetry pass-through

    // Default constructor (required for pre-allocating the ring buffer array)
    StrategyUpdate() : spread_bps(2.0), skew_bps(0.0), regime_id(0), agent_uncertainty(0.0) {}

    // Parameterized constructor
    StrategyUpdate(double spread, double skew, int regime, double uncertainty)
        : spread_bps(spread), skew_bps(skew), regime_id(regime), agent_uncertainty(uncertainty) {}
};

class SPSCQueue {
private:
    std::vector<StrategyUpdate> buffer_;
    const size_t capacity_;

    // head_ is written to by the Python Producer
    alignas(64) std::atomic<size_t> head_;

    // tail_ is written to by the C++ Consumer
    alignas(64) std::atomic<size_t> tail_;

public:
    explicit SPSCQueue(size_t capacity) : buffer_(capacity), capacity_(capacity), head_(0), tail_(0) {}

    // Called by the Python Slow Loop (Orchestrator)
    bool push(const StrategyUpdate& update) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t next_head = (current_head + 1) % capacity_;

        // Memory order acquire ensures we see the latest tail_ written by the consumer
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false; // Queue is full, drop the update (Slow loop is too fast)
        }

        buffer_[current_head] = update;

        // Memory order release ensures the payload is fully written before the head increments
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    // Called by the C++ Fast Loop (Execution Engine)
    std::optional<StrategyUpdate> pop() {
        size_t current_tail = tail_.load(std::memory_order_relaxed);

        // Memory order acquire ensures we see the latest head_ written by the producer
        if (current_tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt; // Queue is empty, nothing to update
        }

        StrategyUpdate update = buffer_[current_tail];

        // Memory order release ensures we finished reading before the tail increments
        tail_.store((current_tail + 1) % capacity_, std::memory_order_release);
        return update;
    }
};
