import time

import numpy as np

import cpp_core


def run_performance_benchmark(iterations: int = 100000):
    print(f"Initializing L1 Hot-Path Benchmark ({iterations} iterations)...")

    # 1. Initialize the lock-free bridge and components
    queue = cpp_core.SPSCQueue(1024)
    engine = cpp_core.LOBEngine(queue, max_pos=1000.0, max_spread=20.0)

    # NEW: OS-Level Hardware Sympathy. Pin the execution engine to CPU Core 0
    engine.pin_thread_to_core(0)

    # Pre-seed the queue with a strategy update from our "Slow Loop"
    initial_update = cpp_core.StrategyUpdate(3.5, -0.5, 0, 0.0)
    queue.push(initial_update)

    latency_timestamps = np.zeros(iterations)

    # Market simulation parameters
    base_price = 150.00
    base_volume = 10.0

    print("Executing Hot Path...")
    for i in range(iterations):
        # Simulate Level-1 Order Book Dynamics
        mid_tick = base_price + (i % 100) * 0.01

        # Fluctuate prices slightly around the mid
        bid_price = mid_tick - 0.01
        ask_price = mid_tick + 0.01

        # Oscillate volumes to trigger the Order Flow Imbalance (OFI) math
        bid_qty = base_volume + (i % 5) * 2.0
        ask_qty = base_volume + (i % 7) * 1.5

        # Periodic asynchronous parameter injection
        if i % 5000 == 0:
            dynamic_update = cpp_core.StrategyUpdate(
                spread_bps=4.0 + (i % 3),
                skew_bps=0.5 if i % 2 == 0 else -0.5,
                regime_id=int((i / 5000) % 4),
                agent_uncertainty=0.0,
            )
            queue.push(dynamic_update)

        start_time = time.perf_counter_ns()

        # CRITICAL CALL: Trigger the C++ execution path with full L1 Data
        engine.on_market_tick(bid_price, bid_qty, ask_price, ask_qty)

        end_time = time.perf_counter_ns()
        latency_timestamps[i] = (end_time - start_time) / 1000.0

    print("-" * 50)
    print("BENCHMARK RESULTS (Thread-Pinned C++ Fast Loop)")
    print("-" * 50)

    # Calculate metrics, discarding the first 100 iterations
    hot_latencies = latency_timestamps[100:]
    p50 = np.percentile(hot_latencies, 50)
    p95 = np.percentile(hot_latencies, 95)
    p99 = np.percentile(hot_latencies, 99)
    avg = np.mean(hot_latencies)

    print(f"Total Ticks Processed : {iterations}")
    print(f"Average Latency       : {avg:.4f} μs")
    print(f"p50 (Median Latency)  : {p50:.4f} μs")
    print(f"p95 Latency           : {p95:.4f} μs")
    print(f"p99 (Tail Latency)    : {p99:.4f} μs")
    print("-" * 50)


if __name__ == "__main__":
    run_performance_benchmark()
