import asyncio
import logging
import random
import time

import numpy as np
import uvloop
from agents.llm_classifier import MacroRegimeClassifier
from agents.rl_jax_agent import JAXRLAgent
from orchestrator.arbiter import StrategyArbiter

import cpp_core

# Enforce strict logging for our audit trail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S.000",
)


async def simulate_lob_ingestion(agent: JAXRLAgent, arbiter: StrategyArbiter):
    """
    The High-Frequency Task: Simulates reading memory-mapped LOB data,
    calculating RL policy updates, and pushing to the C++ bridge.
    """
    logging.info("Started High-Frequency LOB Task.")

    # Simulating a state vector (e.g., bid-ask volume imbalance, momentum)
    state_dim = agent.state_dim

    while True:
        # 1. Simulate reading the next LOB state (In reality, mapped via Apache Arrow)
        dummy_state = np.random.randn(state_dim).tolist()

        # 2. XLA-Compiled RL Inference
        rl_spread, rl_skew = agent.select_action(dummy_state)

        # 3. Push through the Arbiter Matrix to C++
        arbiter.process_signals(rl_spread, rl_skew)

        # Simulate a 10ms LOB micro-tick rate
        await asyncio.sleep(0.01)


async def simulate_macro_news(
    classifier: MacroRegimeClassifier, arbiter: StrategyArbiter
):
    """
    The Low-Frequency Task: Simulates unstructured news dropping asynchronously.
    The LLM processes this on a separate thread pool to avoid blocking the LOB Task.
    """
    logging.info("Started Low-Frequency Macro News Task.")

    mock_headlines = [
        "Federal Reserve announces surprise 50bps rate cut.",
        "Unemployment holds steady, markets flat.",
        "Major geopolitical escalation in the Middle East.",
        "Tech sector rallies on blowout earnings reports.",
    ]

    while True:
        # Wait a random amount of time for breaking news (3 to 8 seconds)
        await asyncio.sleep(random.uniform(3.0, 8.0))

        headline = random.choice(mock_headlines)
        logging.info(f"BREAKING NEWS: {headline}")

        # This await delegates the heavy LLM compute to a background thread
        start_time = time.perf_counter()
        llm_signal = await classifier.classify_headline(headline)
        latency = (time.perf_counter() - start_time) * 1000

        logging.info(f"LLM Inference Complete in {latency:.2f}ms. Signal: {llm_signal}")

        # Pass the LLM classification to the Arbiter to update the risk matrix
        arbiter.update_macro_regime(llm_signal)


async def run_orchestrator():
    """Bootstraps the dependencies and initializes the uvloop concurrency."""
    logging.info("Initializing Optiver MVP Orchestrator...")

    # 1. Initialize the C++ Bridge (Capacity of 1024 updates)
    queue = cpp_core.SPSCQueue(1024)

    # 2. Initialize the Python Orchestrator Components
    arbiter = StrategyArbiter(queue)
    rl_agent = JAXRLAgent(state_dim=10, action_dim=2)

    logging.info("Loading LLM into Metal GPU...")
    # Update to match the exact downloaded filename
    llm_classifier = MacroRegimeClassifier(
        model_path="./data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    )

    # 3. Launch decoupled asynchronous tasks
    lob_task = asyncio.create_task(simulate_lob_ingestion(rl_agent, arbiter))
    news_task = asyncio.create_task(simulate_macro_news(llm_classifier, arbiter))

    try:
        await asyncio.gather(lob_task, news_task)
    except asyncio.CancelledError:
        logging.info("Orchestrator shutting down gracefully.")


if __name__ == "__main__":
    # Inject uvloop for maximum performance
    uvloop.install()

    try:
        asyncio.run(run_orchestrator())
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
