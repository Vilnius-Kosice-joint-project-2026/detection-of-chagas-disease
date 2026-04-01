from pathlib import Path
import pandas as pd
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Sami-trop"


def load_metadata(path=None):
    if path is None:
        path = DATA_DIR / "exams.csv"

    df = pd.read_csv(path)
    df["chagas"] = 1
    df["dataset"] = "samitrop"

    return df


def load_ecg(index, hdf5_path=None):
    if hdf5_path is None:
        hdf5_path = DATA_DIR / "exams.hdf5"

    with h5py.File(hdf5_path, "r") as f:
        tracings = f["tracings"]
        ecg = np.array(tracings[index])

    return ecg


def get_ecg_by_index(df, index, hdf5_path=None):

    row = df.iloc[index]
    ecg = load_ecg(index=index, hdf5_path=hdf5_path)

    return ecg, row


def get_ecg_by_exam_id(df, exam_id, hdf5_path=None):
    row_match = df[df["exam_id"] == exam_id]

    index = row_match.index[0]
    row = row_match.iloc[0]
    ecg = load_ecg(index=index, hdf5_path=hdf5_path)

    return ecg, row