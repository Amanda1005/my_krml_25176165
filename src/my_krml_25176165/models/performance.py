from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def print_regressor_scores(y_preds, y_actuals, set_name="Set"):
    """Print RMSE and MAE for regression predictions."""
    rmse = np.sqrt(mean_squared_error(y_actuals, y_preds))
    mae = mean_absolute_error(y_actuals, y_preds)
    print(f"RMSE {set_name}: {rmse}")
    print(f"MAE {set_name}: {mae}")

# --- Additions for classification tasks ---
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def print_classifier_scores(y_preds, y_actuals, set_name="Set") -> None:
    """Print Accuracy and (weighted) F1 for classification models."""
    acc = accuracy_score(y_actuals, y_preds)
    f1 = f1_score(y_actuals, y_preds, average="weighted", zero_division=0)
    print(f"{set_name} Accuracy: {acc:.4f}")
    print(f"{set_name} F1 (weighted): {f1:.4f}")

def assess_classifier_set(model, features, target, set_name="Set", save_dir: Path | None = None) -> None:
    """
    Predict on a set, save predictions as CSV, and print Accuracy/F1.
    Output file: <save_dir or ./models>/<set_name>_preds.csv
    """
    y_preds = model.predict(features)

    out_dir = save_dir or (Path.cwd() / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{set_name}_preds.csv"

    if isinstance(features, pd.DataFrame):
        pd.DataFrame({"y_pred": y_preds}, index=features.index).to_csv(out_path)
    else:
        pd.DataFrame({"y_pred": y_preds}).to_csv(out_path, index=False)

    print(f"[{set_name}] Saved predictions -> {out_path}")
    print_classifier_scores(y_preds, target, set_name=set_name)

def fit_assess_classifier(model, X_train, y_train, X_val, y_val) -> None:
    """
    Fit on train, then report Accuracy/F1 on train & val.
    """
    model.fit(X_train, y_train)
    print(" Model fitted.")
    assess_classifier_set(model, X_train, y_train, set_name="train")
    assess_classifier_set(model, X_val,   y_val,   set_name="val")
