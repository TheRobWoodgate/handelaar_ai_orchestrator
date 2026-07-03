import time

import numpy as np

import cpp_core


def run_performance_benchmark(iterations: int = 100000):
    print(f"Initializing Hot-Path Latency Benchmark ({iterations} iterations)...")

    # 1. Initialize the lock-free bridge and components
    queue = cpp_core.SPSCQueue(1024)
    engine = cpp_core.LOBEngine(queue, max_pos=1000.0, max_spread=20.0)

    # Pre-seed the queue with a strategy update from our "Slow Loop"
    initial_update = cpp_core.StrategyUpdate(3.5, -0.5, 0)
    queue.push(initial_update)

    # Array to allocate memory for latency collections (in microseconds)
    latency_timestamps = np.zeros(iterations)

    # Simulate a dynamic mid-price moving across ticks
    base_price = 150.00

    print("Executing Hot Path...")
    # 2. Main execution benchmark loop
    for i in range(iterations):
        # Micro-oscillate the mid-price to simulate live LOB ticks
        mid_tick = base_price + (i % 100) * 0.01

        # Periodic asynchronous parameter injection to stress-test the SPSC pop path
        if i % 5000 == 0:
            # Simulate the Python loop updating parameters mid-flight
            dynamic_update = cpp_core.StrategyUpdate(
                spread_bps=4.0 + (i % 3),
                skew_bps=0.5 if i % 2 == 0 else -0.5,
                regime_id=int((i / 5000) % 4),
            )
            queue.push(dynamic_update)

        # Start high-resolution timer
        start_time = time.perf_counter_ns()

        # CRITICAL CALL: Trigger the compiled C++ execution hot path
        engine.on_market_tick(mid_tick)

        # Stop high-resolution timer
        end_time = time.perf_counter_ns()

        # Convert nanoseconds to microseconds
        latency_timestamps[i] = (end_time - start_time) / 1000.0

    print("-" * 50)
    print("BENCHMARK RESULTS (C++ Fast Loop under AI Orchestration)")
    print("-" * 50)

    # 3. Calculate metrics, discarding the first 100 iterations to eliminate cold-start cache misses
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
