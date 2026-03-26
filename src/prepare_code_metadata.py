import pandas as pd


def prepare_code_metadata(
    exams_path: str = "data/CODE-15%/exams.csv",
    labels_path: str = "data/CODE-15%/code15_chagas_labels.csv",
    output_path: str = "data/CODE-15%/code15_merged.csv",
) -> pd.DataFrame:
    exams = pd.read_csv(exams_path)
    labels = pd.read_csv(labels_path)

    df = exams.merge(labels, on="exam_id")

    if "patient_id_x" in df.columns and "patient_id_y" in df.columns:
        same_patient_ids = (df["patient_id_x"] == df["patient_id_y"]).all()
        print("patient_id_x == patient_id_y for all rows:", same_patient_ids)

        if same_patient_ids:
            df = df.drop(columns=["patient_id_y"])
            df = df.rename(columns={"patient_id_x": "patient_id"})

    print("Merged shape:", df.shape)
    print("\nChagas distribution:")
    print(df["chagas"].value_counts(dropna=False))

    df.to_csv(output_path, index=False)
    print(f"\nSaved merged metadata to: {output_path}")

    return df


if __name__ == "__main__":
    prepare_code_metadata()