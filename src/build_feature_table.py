import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from read_code15 import load_metadata, get_ecg_from_row
from preprocessing import preprocess_ecg_record
from feature_extraction import (
    extract_features_from_lead,
    extract_interval_features_from_lead,
    detect_r_peaks,
    compute_sdnn_pnn50,
)


METADATA_COLS = [
    "age", "is_male", "RBBB", "1dAVb", "LBBB", "AF", "SB", "ST"
]

#Script applies preprocessing and feature extraction to all records
#It processes records in batches and saves results to a CSV file.
def choose_interval_lead(usable_leads):
    if 1 in usable_leads:
        return 1
    return usable_leads[0]


def aggregate_lead_features(lead_feature_list):
    lead_df = pd.DataFrame(lead_feature_list)

    aggregated = {}

    for col in lead_df.columns:
        aggregated[f"{col}_mean"] = lead_df[col].mean(skipna=True)
        aggregated[f"{col}_std"] = lead_df[col].std(skipna=True)

    return aggregated


def extract_multivariate_sdnn_pnn50(processed_ecg, usable_leads, fs=400):
    """
    Extract SDNN and pNN50 using RR intervals from all usable leads.
    """
    all_rr_intervals = []

    for lead_idx in usable_leads:
        filtered_lead = processed_ecg["filtered"][:, lead_idx]
        peaks = detect_r_peaks(filtered_lead, fs=fs)

        if len(peaks) > 1:
            rr = np.diff(peaks) / fs * 1000
            all_rr_intervals.extend(rr)

    if len(all_rr_intervals) < 2:
        return {"sdnn": np.nan, "pnn50": np.nan}

    return compute_sdnn_pnn50(np.array(all_rr_intervals))


def extract_features_from_record(row, fs=400):
    try:
        ecg = get_ecg_from_row(row)
        processed = preprocess_ecg_record(ecg, fs=fs)

        if processed is None:
            return None

        usable_leads = processed["usable_leads"]

        if len(usable_leads) == 0:
            return None

        lead_features = []

        for lead_idx in usable_leads:
            filtered_lead = processed["filtered"][:, lead_idx]
            normalized_lead = processed["normalized"][:, lead_idx]

            feats = extract_features_from_lead(
                filtered_lead=filtered_lead,
                normalized_lead=normalized_lead,
                fs=fs,
                include_intervals=False,
            )

            lead_features.append(feats)

        if len(lead_features) == 0:
            return None

        features = aggregate_lead_features(lead_features)

        interval_lead_idx = choose_interval_lead(usable_leads)
        interval_filtered_lead = processed["filtered"][:, interval_lead_idx]

        interval_features = extract_interval_features_from_lead(
            interval_filtered_lead,
            fs=fs,
        )

        features.update(interval_features)
        features.update(extract_multivariate_sdnn_pnn50(processed, usable_leads, fs=fs))

        features["interval_lead_idx"] = interval_lead_idx
        features["n_usable_leads"] = len(usable_leads)

        return features

    except Exception:
        return None


def process_one_row(row, fs=400):
    feats = extract_features_from_record(row, fs=fs)

    if feats is None:
        return None

    feats["exam_id"] = row["exam_id"]
    feats["patient_id"] = row["patient_id"]
    feats["chagas"] = row["chagas"]

    for col in METADATA_COLS:
        if col in row.index:
            feats[col] = row[col]

    return feats


def build_feature_table(
    save_path="new_df.csv",
    fs=400,
    n_jobs=4,
    batch_size=5000,
):
    df = load_metadata()
    df_selected = df.copy().reset_index(drop=True)

    first_batch = True
    total_usable = 0
    total_dropped = 0

    for start in range(0, len(df_selected), batch_size):
        end = min(start + batch_size, len(df_selected))
        batch = df_selected.iloc[start:end]

        rows = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_one_row)(row, fs=fs)
            for _, row in tqdm(batch.iterrows(), total=len(batch))
        )

        rows = [r for r in rows if r is not None]

        total_usable += len(rows)
        total_dropped += len(batch) - len(rows)

        if len(rows) > 0:
            batch_df = pd.DataFrame(rows)

            batch_df.to_csv(
                save_path,
                mode="w" if first_batch else "a",
                header=first_batch,
                index=False
            )

            first_batch = False

        print(f"Processed {end}/{len(df_selected)}")
        print(f"Usable so far: {total_usable}")
        print(f"Dropped so far: {total_dropped}")

    print("Input records:", len(df_selected))
    print("Usable records:", total_usable)
    print("Dropped records:", total_dropped)


if __name__ == "__main__":
    build_feature_table(
        save_path="new_df.csv",
        fs=400,
        n_jobs=4,
    )

