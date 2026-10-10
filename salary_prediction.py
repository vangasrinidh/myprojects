"""Task 5: Demonstration salary prediction with Linear Regression and MSE."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

def main():
    # Clearly labeled synthetic teaching data; salary values are illustrative, not market estimates.
    rng = np.random.default_rng(21)
    years = np.arange(0, 16, dtype=float)
    salary = 30000 + years * 4500 + rng.normal(0, 7000, len(years))
    df = pd.DataFrame({"years_experience": years, "annual_salary_demo": salary.round(0).astype(int)})
    X_train, X_test, y_train, y_test = train_test_split(
        df[["years_experience"]], df["annual_salary_demo"], test_size=0.25, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    predicted = model.predict(X_test)
    mse = mean_squared_error(y_test, predicted)
    print(f"Mean Squared Error: {mse:,.0f} (salary-units squared)")
    print(f"RMSE for easier interpretation: {np.sqrt(mse):,.0f} salary units")
    print(f"Model: salary = {model.intercept_:,.0f} + {model.coef_[0]:,.0f} × years of experience")
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "illustrative_salary_data.csv", index=False)
    plt.scatter(X_test, y_test, label="Actual test examples", color="#2563eb")
    order = np.argsort(X_test["years_experience"].to_numpy())
    plt.plot(X_test["years_experience"].to_numpy()[order], predicted[order], color="#dc2626", label="Predicted")
    plt.xlabel("Years of experience")
    plt.ylabel("Illustrative annual salary")
    plt.title("Simple salary prediction (synthetic demonstration data)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "salary_prediction.png", dpi=160)
    print(f"Saved chart and demonstration CSV in {OUT}")

if __name__ == "__main__":
    main()
