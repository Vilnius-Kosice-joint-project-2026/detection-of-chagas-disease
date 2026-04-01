from pathlib import Path
import pandas as pd
import wfdb
import ast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "PTB-XL"


def load_metadata(path=None):
    if path is None:
        path = DATA_DIR / "ptbxl_database.csv"

    df = pd.read_csv(path)
    df["ecg_path"] = df["filename_hr"]
    df["dataset"] = "ptbxl"

    if "scp_codes" in df.columns:
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    return df


def load_ecg(record_path, base_path=None):
    if base_path is None:
        base_path = DATA_DIR

    full_path = Path(base_path) / record_path
    signal, meta = wfdb.rdsamp(str(full_path))

    return signal, meta