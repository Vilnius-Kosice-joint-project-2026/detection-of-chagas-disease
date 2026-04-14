import numpy as np
import pywt


def apply_wavelet_transform(
    ecg_lead: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
) -> list[np.ndarray]:
    """Apply discrete wavelet transform (DWT) to an ECG lead.

    Args:
        ecg_lead: 1D numpy array of ECG signal.
        wavelet: Wavelet family to use (default 'db4' — standard for ECG).
        level: Decomposition level (default 5).

    Returns:
        List of coefficient arrays [cA_n, cD_n, ..., cD_1]
        (approximation + detail coefficients, coarsest to finest).
    """
    if ecg_lead.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {ecg_lead.shape}")

    max_level = pywt.dwt_max_level(len(ecg_lead), wavelet)
    level = min(level, max_level)

    coeffs = pywt.wavedec(ecg_lead, wavelet=wavelet, level=level)
    return coeffs


def extract_energy_per_level(coeffs: list[np.ndarray]) -> dict:
    """Compute signal energy for each wavelet decomposition level.

    Energy is defined as the sum of squared coefficients.

    Args:
        coeffs: Output of apply_wavelet_transform — list [cA_n, cD_n, ..., cD_1].

    Returns:
        Dict with keys 'energy_approx' and 'energy_detail_1' ... 'energy_detail_N'.
    """
    result = {}
    result["energy_approx"] = float(np.sum(coeffs[0] ** 2))
    for i, detail in enumerate(coeffs[1:], start=1):
        result[f"energy_detail_{i}"] = float(np.sum(detail ** 2))
    return result


def extract_wavelet_features(
    ecg_lead: np.ndarray,
    fs: int = 400,
    wavelet: str = "db4",
    level: int = 5,
) -> dict:
    """Extract all wavelet-based features from one ECG lead.

    Args:
        ecg_lead: 1D numpy array of ECG signal.
        fs: Sampling frequency in Hz (reserved for future band mapping).
        wavelet: Wavelet family (default 'db4').
        level: Decomposition level (default 5).

    Returns:
        Dict with energy per wavelet level (approximation + details).
    """
    coeffs = apply_wavelet_transform(ecg_lead, wavelet=wavelet, level=level)
    return extract_energy_per_level(coeffs)
