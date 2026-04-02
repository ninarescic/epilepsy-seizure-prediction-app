import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

from pipeline.config import AppConfig
from pipeline.preprocessing import validate_preprocessing_outputs
from pipeline.feature_selection import (
    load_all_window_features,
    inspect_feature_table,
    run_feature_selection,
)
from pipeline.models import run_loso_model_training, HAS_XGBOOST

st.set_page_config(page_title="EEG Seizure Prediction App", layout="wide")

st.title("EEG Seizure Prediction App")
st.write("Mini app for seizure prediction on EEG data.")

config = AppConfig()

st.sidebar.header("Paths")
preprocessing_dir = st.sidebar.text_input(
    "Preprocessing output folder",
    str(config.preprocessing_dir)
)

feature_selection_dir = st.sidebar.text_input(
    "Feature selection output folder",
    str(config.feature_selection_dir)
)

models_dir = st.sidebar.text_input(
    "Model output folder",
    str(config.models_dir)
)

if "preprocessing_status" not in st.session_state:
    st.session_state.preprocessing_status = None

if "feature_selection_result" not in st.session_state:
    st.session_state.feature_selection_result = None

if "feature_table_inspection" not in st.session_state:
    st.session_state.feature_table_inspection = None

if "model_result" not in st.session_state:
    st.session_state.model_result = None

st.header("Step 1: Preprocessing outputs")
st.caption("This step is currently external. The app validates an existing preprocessing output folder.")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Check preprocessing"):
        result = validate_preprocessing_outputs(Path(preprocessing_dir))
        st.session_state.preprocessing_status = result

with col2:
    status = st.session_state.preprocessing_status

    if status is None:
        st.info("No check performed yet.")
    else:
        if not status["exists"]:
            st.error("Folder does not exist.")
        elif status["is_valid"]:
            st.success("Preprocessing outputs are ready ✅")
        else:
            st.warning("Preprocessing outputs are incomplete ⚠️")

if st.session_state.preprocessing_status:
    status = st.session_state.preprocessing_status

    with st.expander("Validation details", expanded=True):
        st.write("**Folder**")
        st.code(status["preprocessing_dir"])

        st.write("**Required files found**")
        if status["found_files"]:
            for f in status["found_files"]:
                st.write(f"- {f}")
        else:
            st.write("None")

        st.write("**Required files missing**")
        if status["missing_files"]:
            for f in status["missing_files"]:
                st.write(f"- {f}")
        else:
            st.write("None")

        st.write("**Participant feature files**")
        if status["participant_feature_files"]:
            st.write(f"Found {status['n_participant_feature_files']} participant file(s).")
            for f in status["participant_feature_files"]:
                st.write(f"- {f}")
        else:
            st.write("No participant feature files found.")

st.divider()

st.header("Step 2: Feature selection")

preprocessing_ok = (
    st.session_state.preprocessing_status is not None
    and st.session_state.preprocessing_status["is_valid"]
)

if preprocessing_ok:
    st.success("Step 1 is valid. Feature selection is available.")

    fs_method = st.selectbox(
        "Feature selection method",
        [
            "f_classif",
            "mutual_info",
        ],
    )

    max_features = st.number_input(
        "Maximum number of selected features",
        min_value=1,
        value=20,
        step=1,
    )

    col_a, col_b, col_c = st.columns([1, 1, 1])

    with col_a:
        if st.button("Preview all_window_features.csv"):
            try:
                df = load_all_window_features(Path(preprocessing_dir))
                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                st.dataframe(df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Preview failed: {e}")

    with col_b:
        if st.button("Inspect columns"):
            try:
                df = load_all_window_features(Path(preprocessing_dir))
                inspection = inspect_feature_table(df)
                st.session_state.feature_table_inspection = inspection
                st.success("Inspection complete.")
            except Exception as e:
                st.error(f"Inspection failed: {e}")

    with col_c:
        if st.button("Run Feature Selection"):
            try:
                result = run_feature_selection(
                    preprocessing_dir=Path(preprocessing_dir),
                    output_dir=Path(feature_selection_dir),
                    method=fs_method,
                    max_features=int(max_features),
                )
                st.session_state.feature_selection_result = result
                st.success("Feature selection completed.")
            except Exception as e:
                st.error(f"Feature selection failed: {e}")

    if st.session_state.feature_table_inspection:
        inspection = st.session_state.feature_table_inspection

        with st.expander("Feature table inspection", expanded=True):
            st.write(f"**Rows:** {inspection['n_rows']}")
            st.write(f"**Columns:** {inspection['n_columns']}")
            st.write(f"**Target column:** {inspection['target_column']}")
            st.write(f"**First feature column:** {inspection['first_feature_column']}")
            st.write(f"**First feature column index:** {inspection['first_feature_index']}")
            st.write(f"**Metadata columns:** {len(inspection['metadata_columns'])}")
            st.write(f"**Candidate feature columns:** {len(inspection['feature_columns'])}")
            st.write(f"**Numeric feature columns:** {len(inspection['numeric_feature_columns'])}")

            st.write("**Metadata columns preview**")
            for c in inspection["metadata_columns"][:20]:
                st.write(f"- {c}")

            st.write("**Numeric feature columns preview**")
            for c in inspection["numeric_feature_columns"][:20]:
                st.write(f"- {c}")

    if st.session_state.feature_selection_result:
        fs_result = st.session_state.feature_selection_result

        with st.expander("Feature selection result", expanded=True):
            st.write(f"**Status:** {fs_result['status']}")
            st.write(f"**Method:** {fs_result['method']}")
            st.write(f"**Target column:** {fs_result['target_column']}")
            st.write(f"**Selected feature count:** {fs_result['n_selected_features']}")

            st.write("**Output files**")
            st.code(fs_result["summary_path"])
            st.code(fs_result["feature_scores_path"])
            st.code(fs_result["selected_features_path"])

            st.write("**Selected features**")
            for f in fs_result["selected_features"]:
                st.write(f"- {f}")

else:
    st.warning("Feature selection is locked until preprocessing outputs are valid.")

st.divider()
st.header("Step 3: Model training")

feature_selection_ok = st.session_state.feature_selection_result is not None

if feature_selection_ok:
    st.success("Feature selection available. Model training uses LOSO cross-validation by subject_id.")

    model_options = ["logreg", "random_forest"]
    if HAS_XGBOOST:
        model_options.append("xgboost")

    model_name = st.selectbox("Model", model_options)
    use_selected = st.checkbox("Use selected features", value=True)

    if st.button("Run LOSO model training"):
        try:
            result = run_loso_model_training(
                preprocessing_dir=Path(preprocessing_dir),
                feature_selection_dir=Path(feature_selection_dir),
                output_dir=Path(models_dir),
                use_selected_features=use_selected,
                model_name=model_name,
            )
            st.session_state.model_result = result
            st.success("LOSO model training completed.")
        except Exception as e:
            st.error(f"Model training failed: {e}")

    if st.session_state.model_result:
        res = st.session_state.model_result
        preds = res["predictions_df"]
        fold_metrics = res["fold_metrics_df"]

        with st.expander("Model results", expanded=True):
            st.write("**Summary metrics**")
            st.json(res["summary_metrics"])

            st.write("**Output files**")
            st.code(res["predictions_path"])
            st.code(res["fold_metrics_path"])
            st.code(res["summary_metrics_path"])
            st.code(res["models_dir"])

            st.write("**Fold metrics**")
            st.dataframe(fold_metrics, use_container_width=True)

            st.write("**Held-out predictions preview**")
            st.dataframe(preds.head(20), use_container_width=True)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(preds["y_true"], preds["y_pred"])

        fig_cm, ax_cm = plt.subplots()
        ax_cm.imshow(cm)
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
        ax_cm.set_yticklabels(["True 0", "True 1"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center")

        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        if preds["y_true"].nunique() > 1:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_prob"])

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr)
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            st.pyplot(fig_roc)
        else:
            st.info("ROC curve is unavailable because the predictions contain only one class.")

else:
    st.warning("Run feature selection before training a model.")