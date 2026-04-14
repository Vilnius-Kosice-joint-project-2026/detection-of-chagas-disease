import neurokit2 as nk
import numpy as np
from read_code15 import get_ecg_from_row as get_code_ecg

import numpy as np
from scipy.signal import butter, filtfilt


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


def bandpass_filter(signal, fs=400, lowcut=0.5, highcut=40.0, order=4):
    """
    Bandpass filter for a single ECG lead.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def filter_ecg_multilead(ecg, fs=400, lowcut=0.5, highcut=40.0, order=4):
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
    return np.std(signal) < std_threshold


def has_large_amplitude_artifact(signal, range_threshold=20.0):
    return (np.max(signal) - np.min(signal)) > range_threshold


def get_usable_leads(ecg):
    """
    Return indices of leads that are not obviously flat or severely corrupted.
    """
    usable = []

    for lead_idx in range(ecg.shape[1]):
        signal = ecg[:, lead_idx]

        if is_lead_flat(signal):
            continue

        if has_large_amplitude_artifact(signal):
            continue

        usable.append(lead_idx)

    return usable


def preprocess_ecg_record(ecg, fs=400):
    """
    Full preprocessing for one ECG record.
    Returns None if the record is unusable.
    """
    trimmed = trim_zero_padding(ecg)

    if trimmed.shape[0] < fs * 3:
        return None

    filtered = filter_ecg_multilead(trimmed, fs=fs)
    usable_leads = get_usable_leads(filtered)

    if len(usable_leads) == 0:
        return None

    return {
        "raw": ecg,
        "trimmed": trimmed,
        "filtered": filtered,
        "usable_leads": usable_leads,
    }