from pathlib import Path

import pandas as pd


DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "railway"
)

DATASET_FILE = DATA_DIR / "Train_details_22122017.csv"


def load_train_data() -> pd.DataFrame:

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    df = pd.read_csv(
        DATASET_FILE,
        dtype=str,
        low_memory=False,
    )

    df = df.fillna("")

    df["SEQ"] = pd.to_numeric(
        df["SEQ"],
        errors="coerce",
    )

    df["Distance"] = pd.to_numeric(
        df["Distance"],
        errors="coerce",
    )

    return df