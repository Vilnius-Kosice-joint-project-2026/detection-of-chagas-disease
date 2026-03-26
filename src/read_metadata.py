import pandas as pd
from pathlib import Path

# Project root = folder above src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_sami_metadata():
    path = DATA_DIR / "Sami-trop" / "exams.csv"
    df = pd.read_csv(path)

    # All Sami-trop patients are Chagas positive
    df["chagas"] = 1

    print("Loaded Sami-trop metadata:", df.shape)
    return df


def load_code_metadata():
    path = DATA_DIR / "CODE-15%" / "code15_merged.csv"
    df = pd.read_csv(path)

    df["chagas"] = df["chagas"].astype(int)

    print("Loaded CODE metadata:", df.shape)
    return df


if __name__ == "__main__":
    sami_df = load_sami_metadata()
    code_df = load_code_metadata()

    print("\nSami columns:", sami_df.columns)
    print("\nCODE columns:", code_df.columns)