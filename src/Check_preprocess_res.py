import matplotlib.pyplot as plt
from read_code15 import load_metadata, get_ecg_by_exam_id
from preprocessing import preprocess_ecg_record

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

#This script is used for checking the preprocessing steps on a single ECG record.
def plot_all(raw, filtered, normalized, lead_idx, exam_id):
    plt.figure(figsize=(12, 4))
    plt.plot(raw[:, lead_idx], label="Raw", alpha=0.5)
    plt.plot(filtered[:, lead_idx], label="Filtered")
    plt.plot(normalized[:, lead_idx], label="Normalized")
    plt.title(f"Lead {LEAD_NAMES[lead_idx]} (exam_id={exam_id})")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    df = load_metadata()
    exam_id = 388231
    ecg, meta = get_ecg_by_exam_id(df, exam_id)
    
    print("Exam ID:", exam_id)
    print("ECG shape:", ecg.shape)
    print("Chagas:", meta["chagas"])
    print("Trace file:", meta["trace_file"])

    result = preprocess_ecg_record(ecg, fs=400)

    if result is None:
        print("ECG record unusable after preprocessing.")
        return

    print("Trimmed shape:", result["trimmed"].shape)
    print("Usable leads:", result["usable_leads"])
    print("Usable lead names:", [LEAD_NAMES[i] for i in result["usable_leads"]])

    #lead_idx is for choosing lead. 4 == aVL
    plot_all(result["raw"], result["filtered"], result["normalized"], lead_idx=4, exam_id=exam_id)


if __name__ == "__main__":
    main()