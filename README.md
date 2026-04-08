# EEG Seizure Prediction App

An interactive dashboard for running an EEG-based seizure detection pipeline without requiring coding.

This application wraps a multi-stage machine learning pipeline into a user-friendly interface, allowing non-technical users to:

- Load preprocessed EEG features
- Run feature selection
- Train and evaluate machine learning models
- Apply HMM-based post-processing and explainability

## Pipeline Overview

1. Preprocessing (external for now)
2. Feature Selection
3. Model Training & Evaluation
4. HMM Explainability

## Tech Stack

- Python
- Streamlit (UI)
- scikit-learn / XGBoost
- Pandas / NumPy

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
