import numpy as np
import neurokit2 as nk
import pywt
from scipy.signal import welch

#Functions for ECG feature extraction
def detect_r_peaks(ecg_lead: np.ndarray, fs: int = 400) -> np.ndarray:
    """
    Detect R-peaks in one ECG lead using NeuroKit2.
    """
    if ecg_lead.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {ecg_lead.shape}")

    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    try:
        _, info = nk.ecg_peaks(ecg_lead, sampling_rate=fs)
        return np.array(info["ECG_R_Peaks"])
    except Exception:
        return np.array([])


def compute_rr_features(peaks: np.ndarray, fs: int = 400) -> dict:
    """
    Compute RR-interval and basic HRV features.
    """
    if len(peaks) < 2:
        return {
            "peak_count": len(peaks),
            "rr_mean": np.nan,
            "rr_std": np.nan,
            "hrv_rmssd": np.nan,
        }

    rr = np.diff(peaks) / fs * 1000

    if len(rr) < 2:
        rmssd = np.nan
    else:
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))

    return {
        "peak_count": len(peaks),
        "rr_mean": float(np.mean(rr)),
        "rr_std": float(np.std(rr)),
        "hrv_rmssd": float(rmssd) if not np.isnan(rmssd) else np.nan,
    }


def extract_peak_features_from_lead(ecg_lead: np.ndarray, fs: int = 400) -> dict:
    """
    Extract peak-based and RR-based features from one ECG lead.
    """
    peaks = detect_r_peaks(ecg_lead, fs=fs)
    return compute_rr_features(peaks, fs=fs)


def _safe_mean(values) -> float:
    """
    Return mean while ignoring NaN values.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan

    return float(np.nanmean(values))


def extract_interval_features_from_lead(ecg_lead: np.ndarray, fs: int = 400) -> dict:
    """
    Extract ECG interval features from one lead using NeuroKit2 delineation.

    Features:
    - pr_interval: P onset to R onset / QRS onset approximation
    - qrs_duration: R onset to R offset
    - qt_interval: R onset to T offset
    """
    default_features = {
        "pr_interval": np.nan,
        "qrs_duration": np.nan,
        "qt_interval": np.nan,
    }

    try:
        _, rpeak_info = nk.ecg_peaks(ecg_lead, sampling_rate=fs)
        rpeaks = rpeak_info["ECG_R_Peaks"]

        if len(rpeaks) < 2:
            return default_features

        _, waves = nk.ecg_delineate(
            ecg_lead,
            rpeaks,
            sampling_rate=fs,
            method="dwt"
        )

        p_onsets = np.asarray(waves.get("ECG_P_Onsets", []), dtype=float)
        r_onsets = np.asarray(waves.get("ECG_R_Onsets", []), dtype=float)
        r_offsets = np.asarray(waves.get("ECG_R_Offsets", []), dtype=float)
        t_offsets = np.asarray(waves.get("ECG_T_Offsets", []), dtype=float)

        n = min(
            len(p_onsets),
            len(r_onsets),
            len(r_offsets),
            len(t_offsets),
        )

        if n == 0:
            return default_features

        p_onsets = p_onsets[:n]
        r_onsets = r_onsets[:n]
        r_offsets = r_offsets[:n]
        t_offsets = t_offsets[:n]

        pr_interval = (r_onsets - p_onsets) / fs * 1000
        qrs_duration = (r_offsets - r_onsets) / fs * 1000
        qt_interval = (t_offsets - r_onsets) / fs * 1000

        return {
            "pr_interval": _safe_mean(pr_interval),
            "qrs_duration": _safe_mean(qrs_duration),
            "qt_interval": _safe_mean(qt_interval),
        }

    except Exception:
        return default_features


def compute_psd(
    ecg_lead: np.ndarray,
    fs: int = 400,
    nperseg: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    """
    if ecg_lead.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {ecg_lead.shape}")

    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    freqs, psd = welch(ecg_lead, fs=fs, nperseg=min(nperseg, len(ecg_lead)))
    return freqs, psd


def extract_total_power(freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Compute total power as the integral of PSD using trapezoidal rule.
    """
    return float(np.trapezoid(psd, freqs))


def extract_dominant_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Find the frequency with the highest power.
    """
    return float(freqs[np.argmax(psd)])


def extract_frequency_features_from_lead(ecg_lead: np.ndarray, fs: int = 400) -> dict:
    """
    Extract frequency-domain features from one ECG lead.
    """
    freqs, psd = compute_psd(ecg_lead, fs=fs)

    return {
        "total_power": extract_total_power(freqs, psd),
        "dominant_frequency": extract_dominant_frequency(freqs, psd),
    }


def apply_wavelet_transform(
    ecg_lead: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
) -> list[np.ndarray]:
    """
    Apply discrete wavelet transform to one ECG lead.
    """
    if ecg_lead.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {ecg_lead.shape}")

    max_level = pywt.dwt_max_level(len(ecg_lead), wavelet)
    level = min(level, max_level)

    coeffs = pywt.wavedec(ecg_lead, wavelet=wavelet, level=level)
    return coeffs


def extract_energy_per_level(coeffs: list[np.ndarray]) -> dict:
    """
    Compute wavelet energy for each decomposition level.
    """
    result = {
        "energy_approx": float(np.sum(coeffs[0] ** 2))
    }

    for i, detail in enumerate(coeffs[1:], start=1):
        result[f"energy_detail_{i}"] = float(np.sum(detail ** 2))

    return result


def extract_wavelet_features_from_lead(
    ecg_lead: np.ndarray,
    fs: int = 400,
    wavelet: str = "db4",
    level: int = 5,
) -> dict:
    """
    Extract wavelet-based features from one ECG lead.
    """
    coeffs = apply_wavelet_transform(ecg_lead, wavelet=wavelet, level=level)
    return extract_energy_per_level(coeffs)

def compute_sdnn_pnn50(rr_intervals: np.ndarray) -> dict:
    """
    Compute SDNN and pNN50 from RR intervals.
    """
    if len(rr_intervals) < 2:
        return {"sdnn": np.nan, "pnn50": np.nan}

    rr = np.asarray(rr_intervals, dtype=float)
    rr = rr[~np.isnan(rr)]

    if len(rr) < 2:
        return {"sdnn": np.nan, "pnn50": np.nan}

    sdnn = float(np.std(rr))

    rr_diff = np.diff(rr)
    pnn50 = 100 * np.sum(np.abs(rr_diff) > 50) / len(rr_diff) if len(rr_diff) > 0 else np.nan

    return {
        "sdnn": sdnn,
        "pnn50": float(pnn50),
    }


def extract_features_from_lead(
    filtered_lead: np.ndarray,
    normalized_lead: np.ndarray,
    fs: int = 400,
    include_intervals: bool = True,
) -> dict:
    """
    Extract all features from one ECG lead.

    Peak/RR and interval features are extracted from the filtered signal.
    Frequency and wavelet features are extracted from the normalized signal.
    """
    features = {}

    features.update(extract_peak_features_from_lead(filtered_lead, fs=fs))

    if include_intervals:
        features.update(extract_interval_features_from_lead(filtered_lead, fs=fs))

    features.update(extract_frequency_features_from_lead(normalized_lead, fs=fs))
    features.update(extract_wavelet_features_from_lead(normalized_lead, fs=fs))

    return features