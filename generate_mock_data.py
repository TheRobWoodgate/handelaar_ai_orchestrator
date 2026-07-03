import json
import os

import numpy as np
import pandas as pd


def generate_data():
    print("Generating mock data artifacts...")
    os.makedirs("data", exist_ok=True)

    # 1. Generate LOB Snapshot (Parquet)
    # Simulates 10,000 rows of 10-dimensional microstructure features (e.g., bid-ask imbalances, momentum)
    lob_features = np.random.randn(10000, 10)
    df_lob = pd.DataFrame(lob_features, columns=[f"feature_{i}" for i in range(10)])

    parquet_path = "data/lob_snapshot.parquet"
    df_lob.to_parquet(parquet_path, engine="pyarrow")
    print(f"Created {parquet_path} ({len(df_lob)} rows)")

    # 2. Generate Macro News (JSON)
    mock_news = [
        {
            "timestamp": "2026-07-03T10:00:01Z",
            "headline": "Federal Reserve announces surprise 50bps rate cut.",
        },
        {
            "timestamp": "2026-07-03T10:15:30Z",
            "headline": "Unemployment holds steady, markets flat.",
        },
        {
            "timestamp": "2026-07-03T10:45:00Z",
            "headline": "Major geopolitical escalation in the Middle East.",
        },
        {
            "timestamp": "2026-07-03T11:00:00Z",
            "headline": "Tech sector rallies on blowout earnings reports.",
        },
    ]

    json_path = "data/macro_news.json"
    with open(json_path, "w") as f:
        json.dump(mock_news, f, indent=4)
    print(f"Created {json_path} ({len(mock_news)} events)")


if __name__ == "__main__":
    generate_data()
