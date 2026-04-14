import numpy as np
from scipy.signal import find_peaks


def detect_peaks(ecg_lead: np.ndarray, fs: int = 400, height=None, distance=None) -> np.ndarray:
    """Detect R-peaks in a single ECG lead.

    Args:
        ecg_lead: 1D numpy array of ECG signal.
        fs: Sampling frequency in Hz (default 400).
        height: Minimum peak height (None = no threshold).
        distance: Minimum samples between peaks (default = 200 ms).

    Returns:
        Indices of detected R-peaks.
    """
    if ecg_lead.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {ecg_lead.shape}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if distance is None:
        distance = int(0.2 * fs)  # minimum 200 ms between peaks
    peaks, _ = find_peaks(ecg_lead, height=height, distance=distance)
    return peaks


def compute_rr_intervals(peaks: np.ndarray, fs: int = 400) -> dict:
    """Compute RR-interval statistics from detected R-peaks.

    Args:
        peaks: Indices of R-peaks (output of detect_peaks).
        fs: Sampling frequency in Hz.

    Returns:
        Dict with keys 'rr_mean' and 'rr_std' (ms), or np.nan if <2 peaks.
    """
    if len(peaks) < 2:
        return {"rr_mean": np.nan, "rr_std": np.nan}
    rr = np.diff(peaks) / fs * 1000  # samples → milliseconds
    return {"rr_mean": float(np.mean(rr)), "rr_std": float(np.std(rr))}


def extract_peak_features(ecg: np.ndarray, fs: int = 400, lead_idx: int = 1) -> dict:
    """Extract R-peak features from one ECG record.

    Args:
        ecg: 2D array of shape (samples, leads).
        fs: Sampling frequency in Hz.
        lead_idx: Which lead to analyse (default 1 = Lead II).

    Returns:
        Dict with 'peak_count', 'rr_mean', 'rr_std'.
    """
    if ecg.ndim != 2:
        raise ValueError(f"Expected 2D array (samples, leads), got shape {ecg.shape}")
    if lead_idx >= ecg.shape[1]:
        raise ValueError(f"lead_idx {lead_idx} out of bounds for {ecg.shape[1]} leads")

    peaks = detect_peaks(ecg[:, lead_idx], fs=fs)  # calculate 1 time
    return {"peak_count": len(peaks), **compute_rr_intervals(peaks, fs=fs)}
