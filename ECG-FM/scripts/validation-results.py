import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ECG-FM/results/valid.csv")

df["epoch"] = range(1, len(df) + 1)

metrics = {
    "AUPRC": "auprc",
    "EM Accuracy": "em_accuracy",
    "F1 Score": "f1",
    "Accuracy": "accuracy",
    "AUROC": "auroc"
}

plt.rcParams.update({
    "font.size": 20,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

colors = plt.get_cmap("tab10")

plt.figure(figsize=(12, 7))
plt.style.use("seaborn-v0_8-whitegrid")

for i, (label, col) in enumerate(metrics.items()):
    plt.plot(df["epoch"], df[col], label=label, color=colors(i), linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend(loc="best")
plt.ylim(0.85, 1)
plt.xlim(0, 50)
plt.tight_layout()
plt.grid(False)

output_file = "validation_metrics.pdf"

plt.savefig(output_file, format="pdf", bbox_inches="tight")
plt.show()