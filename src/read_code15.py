from pathlib import Path
import pandas as pd
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "CODE-15%"


def load_metadata(exams_path=None, labels_path=None):
    if exams_path is None:
        exams_path = DATA_DIR / "exams.csv"
    if labels_path is None:
        labels_path = DATA_DIR / "code15_chagas_labels.csv"

    exams_df = pd.read_csv(exams_path)
    labels_df = pd.read_csv(labels_path)

    df = exams_df.merge(labels_df, on="exam_id", how="inner")

    if "patient_id_x" in df.columns and "patient_id_y" in df.columns:
        same_ids = (df["patient_id_x"] == df["patient_id_y"]).all()
        if same_ids:
            df = df.drop(columns=["patient_id_y"])
            df = df.rename(columns={"patient_id_x": "patient_id"})

    if "chagas" in df.columns:
        df["chagas"] = df["chagas"].astype(int)

    df["dataset"] = "code15"

    return df


def load_ecg(exam_id, trace_file, base_path=None):
    if base_path is None:
        base_path = DATA_DIR

    hdf5_path = Path(base_path) / trace_file

    with h5py.File(hdf5_path, "r") as f:
        exam_ids = f["exam_id"][:]
        idx = np.where(exam_ids == exam_id)[0]
        ecg = np.array(f["tracings"][idx[0]])

    return ecg


def get_ecg_by_exam_id(df, exam_id, base_path=None):
    row = df[df["exam_id"] == exam_id]
    row = row.iloc[0]
    ecg = load_ecg(
        exam_id=row["exam_id"],
        trace_file=row["trace_file"],
        base_path=base_path,
    )

    return ecg, row