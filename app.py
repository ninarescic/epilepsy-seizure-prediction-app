import streamlit as st
from pathlib import Path

from pipeline.config import AppConfig
from pipeline.preprocessing import validate_preprocessing_outputs

st.set_page_config(page_title="EEG Seizure Prediction App", layout="wide")

st.title("EEG Seizure Prediction App")
st.write("Mini app for seizure prediction on EEG data.")

config = AppConfig()

st.sidebar.header("Paths")
preprocessing_dir = st.sidebar.text_input(
    "Preprocessing output folder",
    str(config.preprocessing_dir)
)

if "preprocessing_status" not in st.session_state:
    st.session_state.preprocessing_status = None

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

        st.write("**Optional files found**")
        if status["optional_found"]:
            for f in status["optional_found"]:
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

if st.session_state.preprocessing_status and st.session_state.preprocessing_status["is_valid"]:
    st.header("Step 2: Feature selection")
    st.success("Step 1 is valid. We can now add Feature Selection.")
    st.button("Run Feature Selection")
else:
    st.header("Step 2: Feature selection")
    st.warning("Feature selection is locked until preprocessing outputs are valid.")