import numpy as np
from scipy.signal import find_peaks


def detect_peaks(ecg_lead, fs=400, height=None, distance=None):
    """
    Detect R-peaks in a single ECG lead.
    ecg_lead: 1D numpy array
    fs: sampling frequency (default 400 Hz)
    Returns indices of detected peaks.
    """
    if distance is None:
        distance = int(0.2 * fs)  # minimum 200ms between peaks

    peaks, _ = find_peaks(ecg_lead, height=height, distance=distance)
    return peaks


def count_peaks(ecg_lead, fs=400):
    """
    Count number of R-peaks in a single ECG lead.
    """
    peaks = detect_peaks(ecg_lead, fs=fs)
    return len(peaks)


def compute_rr_intervals(ecg_lead, fs=400):
    """
    Compute RR intervals from R-peaks.
    Returns mean and std of RR intervals in milliseconds.
    """
    peaks = detect_peaks(ecg_lead, fs=fs)

    if len(peaks) < 2:
        return {"rr_mean": None, "rr_std": None}

    rr_intervals = np.diff(peaks) / fs * 1000  # convert to milliseconds

    return {
        "rr_mean": float(np.mean(rr_intervals)),
        "rr_std": float(np.std(rr_intervals)),
    }


def extract_peak_features(ecg, fs=400, lead_idx=1):
    """
    Extract all peak features from one ECG record.
    ecg: 2D array (samples, leads)
    lead_idx: which lead to use (default 1 = Lead II)
    """
    lead = ecg[:, lead_idx]

    return {
        "peak_count": count_peaks(lead, fs=fs),
        **compute_rr_intervals(lead, fs=fs),
    }