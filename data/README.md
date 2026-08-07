# Loan approval dataset provenance and quality notes

`loan_approval_dataset.csv` is the final dataset used by this project.

Source page:

- https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

The Kaggle page describes the data as financial records for loan approval
prediction and lists an MIT license. It does not clearly document the original
institution, collection procedure, currency, or whether the records are real or
generated. Consequently, this dataset is appropriate for an educational machine
learning demonstration but not for operational lending decisions.

## Dataset structure

- Rows: 4,269
- Columns: 13
- Target: `loan_status`
- Approved: 2,656 (62.2%)
- Rejected: 1,613 (37.8%)
- Missing values: 0
- Duplicate rows: 0

| Column | Role | Description |
|---|---|---|
| `loan_id` | Identifier | Unique application identifier; excluded from training |
| `no_of_dependents` | Numeric | Number of dependents |
| `education` | Categorical | Graduate or Not Graduate |
| `self_employed` | Categorical | Yes or No |
| `income_annum` | Numeric | Applicant annual income in unspecified dataset units |
| `loan_amount` | Numeric | Requested amount in unspecified dataset units |
| `loan_term` | Numeric | Loan term |
| `cibil_score` | Numeric | CIBIL credit score, ranging from 300 to 900 |
| `residential_assets_value` | Numeric | Residential asset value |
| `commercial_assets_value` | Numeric | Commercial asset value |
| `luxury_assets_value` | Numeric | Luxury asset value |
| `bank_asset_value` | Numeric | Bank asset value |
| `loan_status` | Target | Approved or Rejected |

## Data-quality decisions

- The source headers and categorical values contain leading spaces. The loader
  strips them before validation.
- All `loan_id` values are unique, and the identifier is never used as a model
  feature.
- Twenty-eight rows contain `residential_assets_value = -100000`. The pipeline
  reports these records but preserves them because the source does not state
  whether this value is erroneous or coded information.
- No imputation or SMOTE is applied. The dataset contains no missing values, and
  its class imbalance is moderate. Stratification and precision, recall, and
  F1-score are used instead.
- `cibil_score` is exceptionally predictive of `loan_status`. This limitation
  must be discussed when interpreting high model scores.

## Integrity information

```text
Rows: 4,269
SHA-256: 4B5CD093D178378F4CFA8C107ADB6E599B88BE9D8A3B51F3B99C0D5914154E54
```

`src/prepare_data.py` validates this schema, creates the reproducible train-test
split, and saves a machine-readable data summary under `data/processed/`.
