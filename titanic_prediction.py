"""Task 2: Titanic survival prediction using Logistic Regression."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "train.csv"  # Download from Kaggle; see the project README.
OUT = HERE / "outputs"

def main():
    if not DATA.exists():
        raise FileNotFoundError("Put Kaggle's train.csv in this folder. See the project README.")
    df = pd.read_csv(DATA)
    target = "Survived"
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    missing = set([target, *features]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    X, y = df[features], df[target]
    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical = ["Sex", "Embarked"]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("category", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                                ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("prepare", prep), ("logistic_regression", LogisticRegression(max_iter=1000))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Held-out accuracy: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, target_names=["Did not survive", "Survived"]))
    OUT.mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No", "Yes"], cmap="Blues")
    plt.title("Titanic survival — held-out test split")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=160)
    print(f"Saved confusion matrix to {OUT / 'confusion_matrix.png'}")

if __name__ == "__main__":
    main()
