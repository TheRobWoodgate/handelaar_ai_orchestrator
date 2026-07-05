import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Selenized Dark Color Palette Configuration
# -------------------------------------------------------------------
SLZ_BG = "#103c48"
SLZ_FG = "#adbcbc"
SLZ_DIM = "#184956"
SLZ_RED = "#fa5750"
SLZ_GREEN = "#75b938"
SLZ_BLUE = "#4695f7"
SLZ_CYAN = "#41c7b9"
SLZ_MAGENTA = "#f275be"
SLZ_ORANGE = "#ed8649"

plt.rcParams.update(
    {
        "figure.facecolor": SLZ_BG,
        "axes.facecolor": SLZ_BG,
        "savefig.facecolor": SLZ_BG,
        "text.color": SLZ_FG,
        "axes.labelcolor": SLZ_FG,
        "xtick.color": SLZ_FG,
        "ytick.color": SLZ_FG,
        "axes.edgecolor": SLZ_FG,
        "grid.color": SLZ_DIM,
        "legend.facecolor": SLZ_BG,
        "legend.edgecolor": SLZ_FG,
    }
)


def generate_telemetry_data(ticks: int = 1000) -> pd.DataFrame:
    """Simulates Handelaar telemetry during a flash crash."""
    time = np.arange(ticks)

    # 1. Simulate Market Crash at t=400 to t=500
    mid_price = np.full(ticks, 150.0)
    mid_price[400:500] -= np.linspace(0, 8, 100)  # Crash
    mid_price[500:] = 142.0 + np.random.normal(0, 0.1, ticks - 500)  # Stabilize
    mid_price += np.random.normal(0, 0.05, ticks)

    # 2. Simulate AI Epistemic Uncertainty (Spikes during OOD crash)
    uncertainty = np.random.uniform(0.2, 0.8, ticks)
    uncertainty[410:520] = np.random.normal(5.5, 0.5, 110)  # Spikes above 3.0 threshold

    # 3. Simulate Arbiter Spread (Jumps to 15bps when Uncertainty > 3.0)
    spread = np.where(uncertainty > 3.0, 15.0, np.random.normal(2.0, 0.2, ticks))

    # 4. Simulate OFI and Skew (Lead-Lag)
    ofi = np.sin(time / 20.0) * 10 + np.random.normal(0, 2, ticks)
    skew = np.roll(ofi, shift=5) * 0.1
    skew[:5] = 0.0

    # 5. Simulate Cumulative PnL
    pnl_naive = np.cumsum(np.random.normal(0.01, 0.05, ticks))
    pnl_naive[400:500] -= np.linspace(0, 5, 100)  # Naive gets run over

    pnl_handelaar = np.cumsum(np.random.normal(0.03, 0.04, ticks))
    pnl_handelaar[410:500] = pnl_handelaar[410]  # Handelaar pauses trading

    return pd.DataFrame(
        {
            "tick": time,
            "mid_price": mid_price,
            "ofi": ofi,
            "ai_uncertainty": uncertainty,
            "arbiter_spread_bps": spread,
            "arbiter_skew_bps": skew,
            "pnl_naive": pnl_naive,
            "pnl_handelaar": pnl_handelaar,
        }
    )


def render_visualizations(df: pd.DataFrame):
    """Renders and saves the three interview-ready charts."""
    os.makedirs("assets", exist_ok=True)

    # CHART 1: The Self-Awareness Overlay
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Handelaar Telemetry: Epistemic Kill-Switch Activation",
        fontsize=16,
        color=SLZ_FG,
    )

    ax1.plot(df["tick"], df["mid_price"], color=SLZ_CYAN)
    ax1.set_ylabel("Mid Price ($)")

    ax2.fill_between(df["tick"], df["ai_uncertainty"], color=SLZ_ORANGE, alpha=0.5)
    ax2.axhline(3.0, color=SLZ_RED, linestyle="--", label="Kill-Switch (3.0σ)")
    ax2.set_ylabel("AI Uncertainty (σ)")
    ax2.legend(loc="upper right")

    ax3.step(df["tick"], df["arbiter_spread_bps"], color=SLZ_GREEN)
    ax3.set_ylabel("Arbiter Spread (bps)")
    ax3.set_xlabel("Market Ticks")

    plt.tight_layout()
    # Fix the title overlap by adjusting the top margin after tight_layout
    fig.subplots_adjust(top=0.92)
    plt.savefig("assets/pta_self_awareness.png", dpi=300, bbox_inches="tight")
    plt.close()

    # CHART 2: Cumulative PnL
    plt.figure(figsize=(10, 5))
    plt.plot(
        df["tick"],
        df["pnl_naive"],
        label="Naive Strategy (Static 2bps)",
        color=SLZ_RED,
        alpha=0.8,
    )
    plt.plot(
        df["tick"],
        df["pnl_handelaar"],
        label="Handelaar (RL + Arbiter)",
        color=SLZ_GREEN,
        linewidth=2,
    )
    plt.title("Cumulative PnL: Surviving the Flash Crash", color=SLZ_FG)
    plt.ylabel("Cumulative Profit ($)")
    plt.xlabel("Market Ticks")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/pta_pnl.png", dpi=300, bbox_inches="tight")
    plt.close()

    # CHART 3: OFI Lead-Lag
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    window = df[(df["tick"] > 100) & (df["tick"] < 200)]

    ax1.plot(
        window["tick"],
        window["ofi"],
        color=SLZ_CYAN,
        alpha=0.8,
        label="Order Flow Imbalance (OFI)",
    )
    ax2.plot(
        window["tick"],
        window["arbiter_skew_bps"],
        color=SLZ_MAGENTA,
        linestyle="dashed",
        label="AI Quoted Skew",
    )

    ax1.set_ylabel("OFI (Volume)")
    ax2.set_ylabel("Quoted Skew (bps)")
    ax1.set_xlabel("Market Ticks")
    plt.title("Microstructure Lead-Lag: AI Reacting to OFI Flow", color=SLZ_FG)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    plt.savefig("assets/pta_ofi.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Visualizations successfully saved to /assets/ using Selenized Dark.")


if __name__ == "__main__":
    print("Generating telemetry data...")
    telemetry_df = generate_telemetry_data()

    print("Saving telemetry.parquet...")
    os.makedirs("data", exist_ok=True)
    telemetry_df.to_parquet("data/telemetry.parquet", engine="pyarrow")

    print("Rendering Post-Trade Analytics...")
    render_visualizations(telemetry_df)
