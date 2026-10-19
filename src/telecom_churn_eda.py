"""
Telecom Customer Churn: Exploratory Data Analysis and Deep Learning
Student: Pritheesh Kumar
Course: U21ADP05 - Exploratory Data Analysis and Visualization
Run: python src/telecom_churn_eda.py
The script downloads the public IBM Telco Customer Churn dataset, cleans it,
creates EDA and model-evaluation plots, and saves evaluation metrics.
"""
from pathlib import Path
import urllib.request
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
RANDOM_STATE = 42

def load_data():
    if not DATA_PATH.exists():
        print("Downloading dataset from the public IBM sample repository...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df

def clean_data(df):
    df = df.copy()
    # Convert TotalCharges to numeric; blank strings become missing values.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop_duplicates()
    df = df.dropna(subset=["Churn"])
    # SeniorCitizen is coded 0/1; make it categorical for interpretation.
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"].astype(str))
    return df

def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()

def run_eda(df):
    print("\n--- DATA OVERVIEW ---")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(10))
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nNumeric summary:\n", df.describe(include="all").transpose().head(20))
    print("\nChurn counts:\n", df["Churn"].value_counts())

    # 1. Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Class Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of customers")
    save_plot("01_churn_distribution.png")

    # 2. Tenure distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, element="step")
    plt.title("Customer Tenure Distribution by Churn")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Customer count")
    save_plot("02_tenure_distribution.png")

    # 3. Monthly charges by churn
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly charges")
    save_plot("03_monthly_charges_boxplot.png")

    # 4. Contract type and churn rate
    contract_rate = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    contract_rate.plot(kind="bar", stacked=True, figsize=(8, 4))
    plt.title("Churn Composition within Contract Types")
    plt.xlabel("Contract type")
    plt.ylabel("Percentage of customers")
    plt.legend(title="Churn")
    save_plot("04_contract_churn.png")

    # 5. Correlation heatmap for numeric variables
    numeric = df.select_dtypes(include=np.number)
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Correlation Heatmap of Numeric Features")
    save_plot("05_correlation_heatmap.png")

    # 6. Internet service vs churn
    internet_rate = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
    internet_rate.plot(kind="bar", figsize=(7, 4))
    plt.title("Churn Percentage by Internet Service")
    plt.xlabel("Internet service")
    plt.ylabel("Percentage")
    plt.legend(title="Churn")
    save_plot("06_internet_service_churn.png")

def build_and_evaluate(df):
    # CustomerID is an identifier, not a predictive feature.
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"].map({"No": 0, "Yes": 1})
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Train/validation/test split: validation is used for basic hyperparameter selection.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20,
        random_state=RANDOM_STATE, stratify=y_train_full
    )

    # A small search keeps the assignment reproducible and computationally manageable.
    candidates = [
        {"hidden_layer_sizes": (32, 16), "alpha": 0.0001},
        {"hidden_layer_sizes": (64, 32), "alpha": 0.001},
        {"hidden_layer_sizes": (32,), "alpha": 0.01},
    ]
    best_model, best_score, best_params = None, -1, None
    for params in candidates:
        pipe = Pipeline([
            ("prep", preprocessor),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                alpha=params["alpha"], activation="relu",
                solver="adam", max_iter=250, early_stopping=True,
                validation_fraction=0.15, random_state=RANDOM_STATE
            ))
        ])
        pipe.fit(X_train, y_train)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        print(f"Validation AUC {val_auc:.4f} for {params}")
        if val_auc > best_score:
            best_model, best_score, best_params = pipe, val_auc, params

    # Refit selected configuration on train + validation before final test evaluation.
    final_pipe = Pipeline([
        ("prep", preprocessor),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=best_params["hidden_layer_sizes"],
            alpha=best_params["alpha"], activation="relu",
            solver="adam", max_iter=250, early_stopping=True,
            validation_fraction=0.15, random_state=RANDOM_STATE
        ))
    ])
    final_pipe.fit(X_train_full, y_train_full)
    pred = final_pipe.predict(X_test)
    prob = final_pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, prob)),
        "validation_roc_auc_best": float(best_score),
        "best_parameters": {"hidden_layer_sizes": list(best_params["hidden_layer_sizes"]), "alpha": best_params["alpha"]},
        "test_rows": int(len(y_test)),
        "train_validation_rows": int(len(y_train_full)),
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, pred, target_names=["No Churn", "Churn"], zero_division=0),
        encoding="utf-8"
    )
    print("\n--- TEST METRICS ---")
    print(json.dumps(metrics, indent=2))
    print("\n", classification_report(y_test, pred, target_names=["No Churn", "Churn"], zero_division=0))

    # 7. Confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title("MLP Test Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    save_plot("07_confusion_matrix.png")

    # 8. ROC curve
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"MLP (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.title("ROC Curve on Held-out Test Set")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    save_plot("08_roc_curve.png")

    # 9. Error distribution / predicted probability
    errors = y_test.to_numpy() - pred
    plt.figure(figsize=(6, 4))
    sns.countplot(x=errors)
    plt.title("Classification Error Distribution (Actual - Predicted)")
    plt.xlabel("Actual label minus predicted label")
    plt.ylabel("Number of customers")
    save_plot("09_error_distribution.png")

    # 10. Training loss curve
    loss_curve = final_pipe.named_steps["mlp"].loss_curve_
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve)
    plt.title("MLP Training Loss by Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    save_plot("10_training_loss.png")
    return metrics

def main():
    raw = load_data()
    print("Raw feature count (including target/identifier):", raw.shape[1])
    df = clean_data(raw)
    run_eda(df)
    metrics = build_and_evaluate(df)
    print(f"\nCompleted. Charts: {FIG_DIR}")
    print(f"Metrics: {RESULT_DIR / 'metrics.json'}")
    print("Use the generated metrics and charts to finalize the report.")

if __name__ == "__main__":
    main()
