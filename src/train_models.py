import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def load_data(train_path: Path, test_path: Path):
    """Load training and test datasets and separate features, target, and test IDs."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "Loan_ID" not in test_df.columns or "Loan_Status" not in train_df.columns:
        raise ValueError("Expected 'Loan_ID' in test data and 'Loan_Status' in train data.")

    test_ids = test_df["Loan_ID"].copy()

    train_clean = train_df.drop(columns=["Loan_ID"]).copy()
    test_clean = test_df.drop(columns=["Loan_ID"]).copy()

    y = train_clean["Loan_Status"]
    X = train_clean.drop(columns=["Loan_Status"])

    return X, y, test_clean, test_ids, train_clean


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and categorical preprocessing pipelines inside a ColumnTransformer."""
    num_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_feats),
            ("cat", cat_transformer, cat_feats),
        ],
        remainder="drop",
    )


def build_models(preprocessor: ColumnTransformer, random_state: int = 42) -> dict:
    """Create candidate classification pipelines."""
    svc_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "SVC": svc_pipeline,
        "Random Forest": rf_pipeline,
    }


def evaluate_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate models on a held-out validation split and stratified cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)

    # Determine positive label for binary weighted/macro tracking
    pos_label = "Y" if "Y" in set(y_full.unique()) else y_full.unique()[0]

    summary_rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)

        val_acc = accuracy_score(y_val, val_preds)
        val_f1_pos = f1_score(y_val, val_preds, pos_label=pos_label)
        cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring="accuracy")

        print(f"\n{'=' * 60}")
        print(f"Model: {name}")
        print(f"{'=' * 60}")
        print(f"Validation Accuracy : {val_acc:.4f}")
        print(f"Validation F1 ({pos_label}) : {val_f1_pos:.4f}")
        print(f"5-Fold CV Accuracy  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("\nConfusion Matrix (Validation):")
        print(confusion_matrix(y_val, val_preds))
        print("\nClassification Report (Validation):")
        print(classification_report(y_val, val_preds))

        summary_rows.append(
            {
                "Model": name,
                "Val_Accuracy": round(val_acc, 4),
                f"Val_F1_{pos_label}": round(val_f1_pos, 4),
                "CV_Mean_Acc": round(cv_scores.mean(), 4),
                "CV_Std_Acc": round(cv_scores.std(), 4),
            }
        )

    comparison_df = pd.DataFrame(summary_rows).sort_values(
        by="CV_Mean_Acc", ascending=False
    )
    print(f"\n{'=' * 60}")
    print("Model Comparison Summary (Sorted by 5-Fold CV Accuracy)")
    print(f"{'=' * 60}")
    print(comparison_df.to_string(index=False))

    return comparison_df


def generate_eda_plots(train_clean: pd.DataFrame, output_dir: Path, show: bool = False):
    """Generate basic exploratory plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Loan_Status", data=train_clean)
    plt.title("Loan Approval Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "loan_status_distribution.png", dpi=150)
    if show:
        plt.show()
    plt.close()

    if "ApplicantIncome" in train_clean.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=train_clean)
        plt.title("Applicant Income vs. Loan Status")
        plt.tight_layout()
        plt.savefig(output_dir / "income_vs_loan_status.png", dpi=150)
        if show:
            plt.show()
        plt.close()

def fit_and_save_submissions(
    models: dict,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    output_dir: Path,
):
    """Refit models on full labeled data and write test prediction CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "SVC": "loan_svc_submissions.csv",
        "Random Forest": "loan_rf_submissions.csv",
    }

    for name, model in models.items():
        model.fit(X_full, y_full)
        test_preds = model.predict(X_test)

        sub_df = pd.DataFrame(
            {
                "Loan_ID": test_ids,
                "Loan_Status": test_preds,
            }
        )
        out_path = output_dir / file_map[name]
        sub_df.to_csv(out_path, index=False)
        print(f"Saved {name} predictions to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Loan Eligibility Prediction Pipeline")
    parser.add_argument("--train-file", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test-file", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    X_full, y_full, X_test, test_ids, train_clean = load_data(
        args.train_file, args.test_file
    )

    print(f"Full labeled shape: {X_full.shape} | Test shape: {X_test.shape}")
    print("\nTarget distribution:")
    print(y_full.value_counts())

    if args.save_plots or args.show_plots:
        generate_eda_plots(train_clean, args.output_dir, show=args.show_plots)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=args.test_size,
        stratify=y_full,
        random_state=args.random_state,
    )
    print(f"\nTrain split: {X_train.shape} | Validation split: {X_val.shape}")

    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor, random_state=args.random_state)

    evaluate_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=args.random_state,
    )

    fit_and_save_submissions(
        models=models,
        X_full=X_full,
        y_full=y_full,
        X_test=X_test,
        test_ids=test_ids,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()