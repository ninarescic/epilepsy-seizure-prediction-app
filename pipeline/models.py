from pathlib import Path
import json

import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from pipeline.feature_selection import load_all_window_features, prepare_xy


GROUP_COLUMN = "subject_id"


def load_selected_features(feature_selection_dir: Path):
    path = Path(feature_selection_dir) / "selected_features.json"

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(model_name: str):
    if model_name == "logreg":
        return LogisticRegression(max_iter=1000)

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        if not HAS_XGBOOST:
            raise ValueError("xgboost is not installed in this environment.")
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def safe_roc_auc(y_true, y_prob):
    unique_classes = pd.Series(y_true).nunique()
    if unique_classes < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_predict_proba(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        scores = pd.Series(scores)
        min_s = scores.min()
        max_s = scores.max()
        if max_s == min_s:
            return pd.Series([0.5] * len(scores)).to_numpy()
        return ((scores - min_s) / (max_s - min_s)).to_numpy()

    raise ValueError("Model does not support probability-style output.")


def prepare_model_inputs(X: pd.DataFrame, y: pd.Series):
    X_numeric = X.apply(pd.to_numeric, errors="coerce")

    bad_columns = X_numeric.columns[X_numeric.isna().all()].tolist()
    if bad_columns:
        raise ValueError(
            f"These feature columns became entirely NaN after numeric conversion: {bad_columns[:20]}"
        )

    X_numeric = X_numeric.fillna(X_numeric.median(numeric_only=True))

    y_series = y.copy()

    if y_series.dtype == "object" or str(y_series.dtype).startswith("string"):
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y_series.astype(str)), index=y_series.index)

        class_mapping = {
            str(label): int(idx)
            for idx, label in enumerate(le.classes_)
        }
    else:
        y_encoded = pd.to_numeric(y_series, errors="coerce")

        if y_encoded.isna().any():
            raise ValueError("Target column could not be converted cleanly to numeric.")

        unique_vals = sorted(pd.Series(y_encoded).dropna().unique().tolist())
        class_mapping = {"numeric_values": [int(v) if float(v).is_integer() else float(v) for v in unique_vals]}

    if pd.Series(y_encoded).nunique() < 2:
        raise ValueError("Target column has fewer than 2 classes after encoding.")

    return X_numeric, y_encoded, class_mapping


def to_json_safe(obj):
    if obj is None:
        return None

    if isinstance(obj, (str, bool, int, float)):
        return obj

    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, pd.Series):
        return [to_json_safe(v) for v in obj.tolist()]

    # numpy / pandas scalar fallback
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    return str(obj)


def run_loso_model_training(
    preprocessing_dir: Path,
    feature_selection_dir: Path,
    output_dir: Path,
    use_selected_features: bool,
    model_name: str = "logreg",
):
    preprocessing_dir = Path(preprocessing_dir)
    feature_selection_dir = Path(feature_selection_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_window_features(preprocessing_dir)

    if GROUP_COLUMN not in df.columns:
        raise ValueError(f"Missing LOSO grouping column: {GROUP_COLUMN}")

    X, y, inspection = prepare_xy(df)

    if use_selected_features:
        selected_features = load_selected_features(feature_selection_dir)

        if selected_features is None:
            raise ValueError("Selected features not found. Run feature selection first.")

        missing_selected = [c for c in selected_features if c not in X.columns]
        if missing_selected:
            raise ValueError(f"Selected features missing from X: {missing_selected[:10]}")

        X = X[selected_features]

    X, y, class_mapping = prepare_model_inputs(X, y)

    groups = df[GROUP_COLUMN].copy()
    unique_subjects = sorted(groups.dropna().unique().tolist())

    if len(unique_subjects) < 2:
        raise ValueError("LOSO requires at least 2 unique subjects.")

    base_model = build_model(model_name)

    fold_metrics = []
    prediction_frames = []
    trained_models = {}

    for held_out_subject in unique_subjects:
        train_mask = groups != held_out_subject
        test_mask = groups == held_out_subject

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        if len(X_test) == 0:
            continue

        if pd.Series(y_train).nunique() < 2:
            raise ValueError(
                f"Training fold for held-out subject '{held_out_subject}' has only one class."
            )

        model = clone(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = safe_predict_proba(model, X_test)

        fold_auc = safe_roc_auc(y_test, y_prob)
        fold_acc = float(accuracy_score(y_test, y_pred))

        fold_metrics.append({
            "held_out_subject": str(held_out_subject),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "accuracy": float(fold_acc),
            "roc_auc": None if fold_auc is None else float(fold_auc),
            "positive_rate_test": float(pd.Series(y_test).mean()),
        })

        fold_pred_df = pd.DataFrame({
            "subject_id": groups.loc[test_mask].astype(str).values,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_prob": y_prob,
        })
        prediction_frames.append(fold_pred_df)

        trained_models[str(held_out_subject)] = model

    if not prediction_frames:
        raise ValueError("No LOSO predictions were generated.")

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics)

    overall_accuracy = float(accuracy_score(predictions_df["y_true"], predictions_df["y_pred"]))
    overall_auc = safe_roc_auc(predictions_df["y_true"], predictions_df["y_prob"])

    cm = confusion_matrix(predictions_df["y_true"], predictions_df["y_pred"])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    else:
        tn = fp = fn = tp = None

    summary_metrics = {
        "validation_strategy": "LOSO",
        "group_column": GROUP_COLUMN,
        "model_name": model_name,
        "use_selected_features": bool(use_selected_features),
        "n_subjects": int(len(unique_subjects)),
        "n_total_rows": int(len(df)),
        "n_features_used": int(X.shape[1]),
        "overall_accuracy": float(overall_accuracy),
        "overall_roc_auc": None if overall_auc is None else float(overall_auc),
        "class_mapping": class_mapping,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    summary_metrics = to_json_safe(summary_metrics)

    predictions_path = output_dir / "loso_predictions.csv"
    fold_metrics_path = output_dir / "loso_fold_metrics.csv"
    summary_metrics_path = output_dir / "loso_summary_metrics.json"
    models_dir = output_dir / "models_by_subject"
    models_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(predictions_path, index=False)
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    with open(summary_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    for held_out_subject, model in trained_models.items():
        model_path = models_dir / f"model_excluding_{held_out_subject}.joblib"
        dump(model, model_path)

    return {
        "status": "ok",
        "summary_metrics": summary_metrics,
        "predictions_path": str(predictions_path),
        "fold_metrics_path": str(fold_metrics_path),
        "summary_metrics_path": str(summary_metrics_path),
        "models_dir": str(models_dir),
        "predictions_df": predictions_df,
        "fold_metrics_df": fold_metrics_df,
    }