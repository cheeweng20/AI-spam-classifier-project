# Loan Approval Prediction Using Supervised Machine Learning

This project predicts whether a loan application is **Approved** or **Rejected**
using two supervised classification algorithms:

- Decision Tree as an interpretable single-tree model
- Random Forest as the primary nonlinear model

Both models use the same validated data split and leakage-safe scikit-learn
pipelines. Five-fold stratified cross-validation selects model configurations by
Approved-class F1-score before each selected pipeline is evaluated once on the
untouched test set.

> **Educational use only:** the target records approval decisions, not borrower
> default or repayment. This prototype must not be used for real lending
> decisions.

## Project Structure

```text
loan-approval-prediction/
|-- data/
|   |-- README.md
|   |-- loan_approval_dataset.csv
|   `-- processed/
|-- models/
|-- streamlit_app.py
|-- src/
|   |-- prepare_data.py
|   |-- train_decision_tree.py
|   |-- train_random_forest.py
|   |-- compare_models.py
|   |-- settings.py
|   `-- training_utils.py
|-- tests/
`-- requirements.txt
```

## Dataset

The CSV contains 4,269 applications and 13 columns. `loan_status` is the target,
and `loan_id` is retained for traceability during validation but excluded from
model features. The remaining 11 predictors contain applicant, loan, credit,
and asset information.

The source file contains no missing values or duplicate records. Its original
headers and category values contain leading whitespace, which is normalized on
load. Twenty-eight records have a `residential_assets_value` of `-100000`.
Those values are reported as a quality warning and retained unchanged because
the source does not explain whether they are errors or meaningful codes.

See `data/README.md` for provenance, schema, class counts, integrity information,
and limitations.

## Installation

Install the required packages using the Python interpreter that will run the
project:

```bash
python -m pip install -r requirements.txt
```

## Complete Workflow

Run the following commands from the project root:

```bash
python src/prepare_data.py
python src/train_decision_tree.py
python src/train_random_forest.py
python src/compare_models.py
python -m streamlit run streamlit_app.py
```

The Streamlit interface collects all 11 model features, displays the prediction
from each model, highlights model agreement, and shows the saved four-metric
model-comparison results.

## Data Preparation

`prepare_data.py`:

1. Loads `data/loan_approval_dataset.csv`.
2. Strips whitespace from headers and categorical values.
3. Validates required columns, labels, categories, identifiers, and numeric
   ranges.
4. Reports negative residential-asset values without silently changing them.
5. Removes exact duplicate applications and rejects conflicting labels.
6. Excludes `loan_id` from the predictor matrix.
7. Creates a stratified 70:30 train-test split using random state 42.
8. Confirms no identical application occurs in both splits.
9. Saves the untouched feature and label splits to `data/processed/`.

## Preprocessing and Model Selection

Preprocessing is fitted inside every model pipeline and every cross-validation
fold:

- Decision Tree and Random Forest pass numeric features through unchanged and
  one-hot encode `education` and `self_employed`.

GridSearchCV uses five-fold `StratifiedKFold` validation. It records accuracy,
Approved-class precision, recall, and F1-score. F1-score is the selection metric
because it balances false approvals and false rejections while accounting for
the 62.2%/37.8% class distribution.

## Generated Results

Final performance on the untouched 1,281-row test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 98.44% | 98.26% | 99.25% | 98.75% |
| Decision Tree | 98.05% | 98.01% | 98.87% | 98.44% |

Random Forest produced 20 incorrect predictions: 14 rejected applications were
predicted Approved and 6 approved applications were predicted Rejected.
Decision Tree produced 25 incorrect predictions: 16 rejected applications were
predicted Approved and 9 approved applications were predicted Rejected.

The selected Decision Tree used unrestricted depth, a minimum leaf size of five,
and no class weighting. The selected Random Forest used unrestricted depth, a
minimum leaf size of one, and no class weighting.

The training and comparison scripts create:

- `models/decision_tree_model.joblib`
- `models/random_forest_model.joblib`
- `models/decision_tree_grid_search.csv`
- `models/random_forest_grid_search.csv`
- `models/decision_tree_training_summary.json`
- `models/random_forest_training_summary.json`
- `models/confusion_matrix_decision_tree.png`
- `models/confusion_matrix_random_forest.png`
- `models/comparison_table.csv`
- `models/comparison_chart.png`

The high performance of the tree-based model must be interpreted cautiously.
`cibil_score` is strongly associated with the target, and results from this
single educational dataset should not be generalized to real applicants.

## Testing

```bash
python -m unittest discover -s tests -v
```

The tests cover schema and range validation, anomaly handling, split leakage,
metric definitions, preprocessing pipelines, model-search configuration, and
Streamlit inference.
