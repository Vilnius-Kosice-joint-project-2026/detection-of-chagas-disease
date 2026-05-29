import numpy as np
from scipy.signal import butter, filtfilt

# Functions for ECG preprocessing
def trim_zero_padding(ecg, tol=1e-8):
    """
    Remove rows where all leads are approximately zero.
    ECG shape: (samples, leads)
    """
    mask = np.any(np.abs(ecg) > tol, axis=1)

    if not np.any(mask):
        return ecg

    first = np.argmax(mask)
    last = len(mask) - np.argmax(mask[::-1])
    return ecg[first:last]


def bandpass_filter(signal, fs=400, lowcut=0.5, highcut=60.0, order=2):
    """
    Bandpass filter for a single ECG lead.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def filter_ecg_multilead(ecg, fs=400, lowcut=0.5, highcut=60.0, order=2):
    """
    Apply bandpass filter to each lead separately.
    ECG shape: (samples, leads)
    """
    filtered = np.zeros_like(ecg, dtype=float)

    for lead_idx in range(ecg.shape[1]):
        filtered[:, lead_idx] = bandpass_filter(
            ecg[:, lead_idx],
            fs=fs,
            lowcut=lowcut,
            highcut=highcut,
            order=order,
        )

    return filtered


def normalize_signal(signal, eps=1e-8):
    """
    Apply z-score normalization to one ECG lead.
    """
    mean = np.mean(signal)
    std = np.std(signal)

    if std < eps:
        return signal

    return (signal - mean) / std


def normalize_ecg_multilead(ecg):
    """
    Apply z-score normalization to each lead separately.
    ECG shape: (samples, leads)
    """
    normalized = np.zeros_like(ecg, dtype=float)

    for lead_idx in range(ecg.shape[1]):
        normalized[:, lead_idx] = normalize_signal(ecg[:, lead_idx])

    return normalized


def assess_lead_quality(signal):
    """
    Compute simple quality metrics for one lead.
    """
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "range": float(np.max(signal) - np.min(signal)),
        "diff_std": float(np.std(np.diff(signal))),
    }


def is_lead_flat(signal, std_threshold=1e-3):
    """
    Detect leads with almost no variation.
    """
    return np.std(signal) < std_threshold


def has_large_amplitude_artifact(signal, range_threshold=20.0):
    """
    Detect leads with unrealistically large amplitude range.
    """
    return (np.max(signal) - np.min(signal)) > range_threshold

def has_spike(signal, threshold=10.0):
    return np.max(np.abs(signal)) > threshold

def get_usable_leads(ecg):
    """
    Return indices of leads that are not obviously flat or severely corrupted.
    This should be applied before normalization, because normalization changes amplitude scale.
    """
    usable = []

    for lead_idx in range(ecg.shape[1]):
        signal = ecg[:, lead_idx]

        if is_lead_flat(signal):
            continue

        if has_large_amplitude_artifact(signal):
            continue

        if has_spike(signal):
            continue

        usable.append(lead_idx)

    return usable


def preprocess_ecg_record(ecg, fs=400):
    """
    Full preprocessing for one ECG record.

    Steps:
    1. Remove zero-padding.
    2. Exclude signals shorter than 3 seconds.
    3. Apply 0.5-40 Hz bandpass filter.
    4. Identify usable leads.
    5. Apply z-score normalization per lead.

    Returns None if the record is unusable.
    """
    trimmed = trim_zero_padding(ecg)

    if trimmed.shape[0] < fs * 3:
        return None

    filtered = filter_ecg_multilead(trimmed, fs=fs)

    usable_leads = get_usable_leads(filtered)

    if len(usable_leads) < 2:
        return None

    normalized = normalize_ecg_multilead(filtered)

    return {
        "raw": ecg,
        "trimmed": trimmed,
        "filtered": filtered,
        "normalized": normalized,
        "usable_leads": usable_leads,
    }