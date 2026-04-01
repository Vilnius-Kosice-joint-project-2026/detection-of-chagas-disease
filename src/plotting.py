import matplotlib.pyplot as plt

def plot_ecg(ecg, title="ECG", lead_names=None):
    n_samples, n_leads = ecg.shape
    fig, axes = plt.subplots(n_leads, 1, figsize=(14, 2 * n_leads), sharex=True)

    if n_leads == 1:
        axes = [axes]

    for i in range(n_leads):
        axes[i].plot(ecg[:, i])
        label = lead_names[i] if lead_names is not None else f"Lead {i+1}"
        axes[i].set_title(label)
        axes[i].set_ylabel("Amplitude")

    axes[-1].set_xlabel("Samples")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()