from pathlib import Path
import pandas as pd
import h5py
import numpy as np

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data" / "CODE-15%"

# Helper functions to load metadata and ECG data for the CODE-15% dataset
def load_metadata(exams_path=None, labels_path=None):
    '''
    This function loads the metadata for the CODE-15% dataset by merging the exams and labels CSV files.
    '''
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
    '''
    This function loads the ECG tracing data for a given exam_id from the specified trace_file.
    It reads the HDF5 file, finds the index of the exam_id, and retrieves the corresponding ECG tracing data as a NumPy array.
    '''
    if base_path is None:
        base_path = DATA_DIR

    hdf5_path = Path(base_path) / trace_file

    with h5py.File(hdf5_path, "r") as f:
        exam_ids = f["exam_id"][:]
        idx = np.where(exam_ids == exam_id)[0]
        ecg = np.array(f["tracings"][idx[0]])

    return ecg


def get_ecg_by_exam_id(df, exam_id, base_path=None):
    '''
    This function retrieves the ECG tracing data and corresponding metadata for a given exam_id from the DataFrame.
    '''
    row = df[df["exam_id"] == exam_id]
    row = row.iloc[0]
    ecg = load_ecg(
        exam_id=row["exam_id"],
        trace_file=row["trace_file"],
        base_path=base_path,
    )

    return ecg, row

def get_ecg_from_row(row, base_path=None):
    '''
    This function retrieves the ECG tracing data for a given row of metadata.
    '''
    ecg = load_ecg(
        exam_id=row["exam_id"],
        trace_file=row["trace_file"],
        base_path=base_path,
    )
    return ecg
    