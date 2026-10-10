"""Task 8: Demonstrate null handling, date conversion and outlier treatment."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

def main():
    # Small invented data to demonstrate cleaning steps reproducibly.
    raw = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "02/01/2026", "2026-01-04", "not-a-date", "2026-01-06"],
        "units_sold": [12, 15, None, 17, 16, 900],
        "unit_price": [10.5, 10.5, 11.0, None, 11.0, 10.5],
    })
    OUT.mkdir(exist_ok=True)
    raw.to_csv(OUT / "messy_sales_demo.csv", index=False)
    print("Missing values before cleaning:\n", raw.isna().sum().to_string())
    clean = raw.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce", format="mixed")
    clean["units_sold"] = pd.to_numeric(clean["units_sold"], errors="coerce")
    clean["unit_price"] = pd.to_numeric(clean["unit_price"], errors="coerce")
    clean = clean.dropna(subset=["date"]).copy()  # Invalid date cannot be reliably analyzed.
    clean["units_sold"] = clean["units_sold"].fillna(clean["units_sold"].median())
    clean["unit_price"] = clean["unit_price"].fillna(clean["unit_price"].median())
    # IQR bounds flag unusually low/high unit counts; cap flagged values rather than deleting rows.
    q1, q3 = clean["units_sold"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flags = (clean["units_sold"] < low) | (clean["units_sold"] > high)
    print(f"IQR outlier bounds: {low:.1f} to {high:.1f}; flagged rows: {int(flags.sum())}")
    clean["units_sold"] = clean["units_sold"].clip(lower=low, upper=high)
    clean.to_csv(OUT / "cleaned_sales_demo.csv", index=False, date_format="%Y-%m-%d")
    print("Cleaning steps: invalid dates coerced to missing then removed; numeric columns converted; missing numeric values filled with the median; outlier capped using the IQR bounds.")
    print("Missing values after cleaning:\n", clean.isna().sum().to_string())
    plt.figure(figsize=(7, 4))
    plt.boxplot(clean["units_sold"])
    plt.ylabel("Units sold after IQR treatment")
    plt.title("Sales example after cleaning")
    plt.tight_layout()
    plt.savefig(OUT / "cleaned_units_boxplot.png", dpi=160)
    print(f"Saved before/after CSVs and chart in {OUT}")

if __name__ == "__main__":
    main()
