import numpy as np
from scipy.signal import welch


def compute_psd(ecg_lead: np.ndarray, fs: int = 400, nperseg: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density using Welch's method.

    Args:
        ecg_lead: 1D numpy array of ECG signal.
        fs: Sampling frequency in Hz.
        nperseg: Samples per segment for Welch (default 256).

    Returns:
        Tuple (freqs, psd) — frequency bins and power values.
    """
    if ecg_lead.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {ecg_lead.shape}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    freqs, psd = welch(ecg_lead, fs=fs, nperseg=min(nperseg, len(ecg_lead)))
    return freqs, psd


def extract_total_power(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Compute total power (integral of PSD via trapezoidal rule).

    Args:
        freqs: Frequency bins (Hz).
        psd: Power spectral density values.

    Returns:
        Total power as float.
    """
    return float(np.trapezoid(psd, freqs))


def extract_mean_power(psd: np.ndarray) -> float:
    """Compute mean power across all frequency bins.

    Args:
        psd: Power spectral density values.

    Returns:
        Mean power as float.
    """
    return float(np.mean(psd))


def extract_dominant_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Find the frequency with the highest power.

    Args:
        freqs: Frequency bins (Hz).
        psd: Power spectral density values.

    Returns:
        Dominant frequency in Hz.
    """
    return float(freqs[np.argmax(psd)])


def extract_frequency_features(ecg_lead: np.ndarray, fs: int = 400) -> dict:
    """Extract all PSD-based features from one ECG lead.

    Args:
        ecg_lead: 1D numpy array of ECG signal.
        fs: Sampling frequency in Hz.

    Returns:
        Dict with 'total_power', 'mean_power', 'dominant_frequency'.
    """
    freqs, psd = compute_psd(ecg_lead, fs=fs)
    return {
        "total_power": extract_total_power(freqs, psd),
        "mean_power": extract_mean_power(psd),
        "dominant_frequency": extract_dominant_frequency(freqs, psd),
    }
