"""Task 6: Correlation heatmap for a small synthetic student-score dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

def main():
    rng = np.random.default_rng(8)
    study = rng.integers(1, 11, 40)
    df = pd.DataFrame({
        "study_hours_per_week": study,
        "attendance_percent": np.clip(55 + study * 4 + rng.normal(0, 9, 40), 40, 100).round(1),
        "math_score": np.clip(35 + study * 5 + rng.normal(0, 10, 40), 0, 100).round(1),
        "science_score": np.clip(40 + study * 4 + rng.normal(0, 12, 40), 0, 100).round(1),
    })
    corr = df.corr(numeric_only=True)
    print("Correlation matrix (values range from -1 to +1):")
    print(corr.round(2).to_string())
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "student_scores_demo.csv", index=False)
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, fmt=".2f")
    plt.title("Student study habits and scores (synthetic demonstration data)")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=160)
    print(f"Saved heatmap and demonstration CSV in {OUT}")

if __name__ == "__main__":
    main()
