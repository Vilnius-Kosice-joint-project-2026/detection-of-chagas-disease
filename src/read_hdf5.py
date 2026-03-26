import os
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_sami_ecg(index=0):
    path = DATA_DIR / "Sami-trop" / "exams.hdf5"
    with h5py.File(path, "r") as f:
        ecg = f["tracings"][index]
    return ecg


def load_code_ecg(exam_id, trace_file):
    path = DATA_DIR / "CODE-15%" / trace_file

    with h5py.File(path, "r") as f:
        exam_ids = f["exam_id"][:]
        idx = np.where(exam_ids == exam_id)[0][0]
        ecg = f["tracings"][idx]

    return ecg

def list_hdf5_files(dataset_dir: str) -> list[str]:
    """
    Return sorted list of .hdf5 / .h5 files in a dataset directory.
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    files = sorted(
        [
            str(p)
            for p in dataset_path.iterdir()
            if p.is_file() and p.suffix.lower() in [".hdf5", ".h5"]
        ]
    )

    if not files:
        raise FileNotFoundError(f"No HDF5 files found in: {dataset_dir}")

    return files


def inspect_hdf5_file(file_path: str) -> dict:
    """
    Inspect one HDF5 file and return basic information.
    """
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())

        if "tracings" not in f:
            raise KeyError(f"'tracings' dataset not found in {file_path}")

        tracings = f["tracings"]

        return {
            "file_path": file_path,
            "keys": keys,
            "shape": tracings.shape,
            "dtype": str(tracings.dtype),
        }


def print_hdf5_info(file_path: str) -> None:
    info = inspect_hdf5_file(file_path)
    print(f"\nFile: {info['file_path']}")
    print("Keys:", info["keys"])
    print("Shape:", info["shape"])
    print("Dtype:", info["dtype"])


def inspect_dataset_folder(dataset_dir: str) -> None:
    """
    Print info for all HDF5 files in a dataset folder.
    """
    files = list_hdf5_files(dataset_dir)
    print(f"\nDataset folder: {dataset_dir}")
    print(f"Found {len(files)} HDF5 file(s).")

    for file_path in files:
        try:
            print_hdf5_info(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")


def load_ecg_by_index(file_path: str, index: int = 0) -> np.ndarray:
    """
    Load one ECG by index from one HDF5 file.

    Returns shape:
        (signal_length, num_leads)
    """
    with h5py.File(file_path, "r") as f:
        tracings = f["tracings"]

        if index < 0 or index >= len(tracings):
            raise IndexError(f"Index {index} out of range for file {file_path}")

        ecg = np.array(tracings[index])

    return ecg


def plot_ecg(ecg: np.ndarray, title: str = "ECG") -> None:
    """
    Plot ECG assuming shape is (signal_length, num_leads).
    """
    if ecg.ndim != 2:
        raise ValueError(f"Expected 2D ECG array, got shape {ecg.shape}")

    signal_length, num_leads = ecg.shape

    plt.figure(figsize=(12, 10))
    for i in range(num_leads):
        plt.subplot(6, 2, i + 1)
        plt.plot(ecg[:, i])
        plt.title(f"Lead {i + 1}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    '''
    Change this depending on what you want to inspect
    for example use "data/CODE-15%" or "data/Sami-trop"
    '''
    dataset_dir = "data/Sami-trop"

    inspect_dataset_folder(dataset_dir)

    files = list_hdf5_files(dataset_dir)
    sample_file = files[0]

    ecg = load_ecg_by_index(sample_file, index=0)
    plot_ecg(ecg, title=os.path.basename(sample_file))