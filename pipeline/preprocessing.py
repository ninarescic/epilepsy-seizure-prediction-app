from pathlib import Path


REQUIRED_PREPROCESSING_FILES = [
    "all_window_features.csv",
    "overall_summary.csv",
    "participant_summary.csv",
    "recording_manifest.csv",
]

OPTIONAL_PREPROCESSING_FILES = [
    "processing_errors.csv",
    "README.txt",
]


def validate_preprocessing_outputs(preprocessing_dir: Path) -> dict:
    preprocessing_dir = Path(preprocessing_dir)

    participant_feature_files = []
    found_files = []
    missing_files = []
    optional_found = []
    optional_missing = []

    result = {
        "preprocessing_dir": str(preprocessing_dir),
        "exists": preprocessing_dir.exists(),
        "found_files": found_files,
        "missing_files": missing_files,
        "optional_found": optional_found,
        "optional_missing": optional_missing,
        "participant_feature_files": participant_feature_files,
        "n_participant_feature_files": 0,
        "is_valid": False,
    }

    if not preprocessing_dir.exists():
        return result

    for file_name in REQUIRED_PREPROCESSING_FILES:
        file_path = preprocessing_dir / file_name
        if file_path.exists():
            found_files.append(file_name)
        else:
            missing_files.append(file_name)

    for file_name in OPTIONAL_PREPROCESSING_FILES:
        file_path = preprocessing_dir / file_name
        if file_path.exists():
            optional_found.append(file_name)
        else:
            optional_missing.append(file_name)

    for file_path in preprocessing_dir.glob("*_window_features.csv"):
        if file_path.name != "all_window_features.csv":
            participant_feature_files.append(file_path.name)

    participant_feature_files.sort()
    result["n_participant_feature_files"] = len(participant_feature_files)

    result["is_valid"] = (
        len(missing_files) == 0 and len(participant_feature_files) > 0
    )

    return result