from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1].parent / "data" / "raw"


def load_data(filename: str = "Telecom Customers Churn.csv") -> pd.DataFrame:
    """Load the telecom churn dataset from the raw data directory."""
    path = DATA_DIR / filename
    return pd.read_csv(path)
