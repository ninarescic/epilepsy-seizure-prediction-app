from pathlib import Path
import json

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


FIRST_FEATURE_COLUMN = "line_length__ch01"


def load_all_window_features(preprocessing_dir: Path) -> pd.DataFrame:
    preprocessing_dir = Path(preprocessing_dir)
    features_path = preprocessing_dir / "all_window_features.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing file: {features_path}")

    return pd.read_csv(features_path)


def detect_target_column(df: pd.DataFrame) -> str:
    if "label" in df.columns:
        return "label"
    if "target" in df.columns:
        return "target"
    raise ValueError("Could not find target column. Expected 'label' or 'target'.")


def inspect_feature_table(df: pd.DataFrame) -> dict:
    target_column = detect_target_column(df)

    if FIRST_FEATURE_COLUMN not in df.columns:
        raise ValueError(
            f"Could not find '{FIRST_FEATURE_COLUMN}'. It is required as the first feature column."
        )

    first_feature_idx = df.columns.get_loc(FIRST_FEATURE_COLUMN)

    metadata_columns = df.columns[:first_feature_idx].tolist()
    candidate_columns = df.columns[first_feature_idx:].tolist()

    feature_columns = [c for c in candidate_columns if c != target_column]

    numeric_feature_columns = df[feature_columns].select_dtypes(include=["number"]).columns.tolist()
    non_numeric_feature_columns = [c for c in feature_columns if c not in numeric_feature_columns]

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target_column": target_column,
        "first_feature_column": FIRST_FEATURE_COLUMN,
        "first_feature_index": int(first_feature_idx),
        "metadata_columns": metadata_columns,
        "feature_columns": feature_columns,
        "numeric_feature_columns": numeric_feature_columns,
        "non_numeric_feature_columns": non_numeric_feature_columns,
    }


def prepare_xy(df: pd.DataFrame):
    inspection = inspect_feature_table(df)

    target_column = inspection["target_column"]
    feature_columns = inspection["numeric_feature_columns"]

    if not feature_columns:
        raise ValueError("No numeric feature columns available for feature selection.")

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    return X, y, inspection


def select_features_filter_method(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    max_features: int,
):
    k = min(max_features, X.shape[1])

    if method == "f_classif":
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError(f"Unsupported filter method: {method}")

    selector.fit(X, y)

    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    scores = selector.scores_

    result_df = pd.DataFrame({
        "feature": X.columns,
        "score": scores,
        "selected": selected_mask,
    })

    result_df = result_df.sort_values(
        by=["selected", "score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result_df, selected_features


def run_feature_selection(
    preprocessing_dir: Path,
    output_dir: Path,
    method: str,
    max_features: int,
) -> dict:
    preprocessing_dir = Path(preprocessing_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_window_features(preprocessing_dir)
    X, y, inspection = prepare_xy(df)

    if method not in {"f_classif", "mutual_info"}:
        raise ValueError("Only 'f_classif' and 'mutual_info' are supported right now.")

    scores_df, selected_features = select_features_filter_method(
        X=X,
        y=y,
        method=method,
        max_features=int(max_features),
    )

    summary = {
        "method": method,
        "target_column": inspection["target_column"],
        "first_feature_column": inspection["first_feature_column"],
        "n_rows": int(df.shape[0]),
        "n_columns_total": int(df.shape[1]),
        "n_metadata_columns": len(inspection["metadata_columns"]),
        "n_feature_columns": len(inspection["feature_columns"]),
        "n_numeric_feature_columns": len(inspection["numeric_feature_columns"]),
        "n_selected_features": len(selected_features),
        "selected_features_file": str(output_dir / "selected_features.json"),
        "feature_scores_file": str(output_dir / "feature_scores.csv"),
        "status": "completed",
    }

    summary_df = pd.DataFrame([summary])

    feature_scores_path = output_dir / "feature_scores.csv"
    selected_features_path = output_dir / "selected_features.json"
    summary_path = output_dir / "feature_selection_summary.csv"

    scores_df.to_csv(feature_scores_path, index=False)

    with open(selected_features_path, "w", encoding="utf-8") as f:
        json.dump(selected_features, f, indent=2)

    summary_df.to_csv(summary_path, index=False)

    return {
        "status": "ok",
        "summary_path": str(summary_path),
        "feature_scores_path": str(feature_scores_path),
        "selected_features_path": str(selected_features_path),
        "method": method,
        "target_column": inspection["target_column"],
        "first_feature_column": inspection["first_feature_column"],
        "n_rows": int(df.shape[0]),
        "n_columns_total": int(df.shape[1]),
        "n_metadata_columns": len(inspection["metadata_columns"]),
        "n_feature_columns": len(inspection["feature_columns"]),
        "n_numeric_feature_columns": len(inspection["numeric_feature_columns"]),
        "n_selected_features": len(selected_features),
        "selected_features": selected_features,
        "inspection": inspection,
    }