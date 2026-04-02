Synthetic SWEC-like feature package for fast pipeline testing.

Contents:
- 5 synthetic subjects: ['SIM1', 'SIM2', 'SIM3', 'SIM4', 'SIM5']
- 2 seizure recordings per subject
- 24 windows per recording (12 non_seizure + 12 seizure, 5 s each)
- 100 numeric features per window:
  - line_length__ch01..ch20
  - amp_var__ch01..ch20
  - mean__ch01..ch20
  - max__ch01..ch20
  - min__ch01..ch20

This package is designed to match the notebooks' expected files in derived_features_swec_short/.
