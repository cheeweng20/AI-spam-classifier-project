# Message Spam Classifier - Supervised Machine Learning

This project classifies messages as ham or spam using two supervised
classification algorithms:

- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

Both models are trained and evaluated using the same cleaned training and test
sets. Each model uses a leakage-safe TF-IDF pipeline and reproducible five-fold
cross-validation to select its best configuration.

## Project Structure

```text
AI-spam-classifier-project/
|-- data/
|   |-- README.md
|   |-- messages.csv
|   `-- processed/
|-- models/
|-- streamlit_app.py
|-- src/
|   |-- prepare_data.py
|   |-- train_naive_bayes.py
|   |-- train_svm.py
|   |-- compare_models.py
|   |-- settings.py
|   `-- text_processing.py
|-- tests/
`-- requirements.txt
```

The input file must be `data/messages.csv`. It must contain a message column
and a label column. Labels may be written as `ham`/`spam` or `0`/`1`.

## Installation

Install the required packages using the same Python interpreter that will run
the project:

```bash
python -m pip install -r requirements.txt
```

## Run the Complete Workflow

Run these commands from the project root in the following order:

```bash
python src/prepare_data.py
python src/train_naive_bayes.py
python src/train_svm.py
python src/compare_models.py
python -m streamlit run streamlit_app.py
```

The Streamlit interface provides two-model prediction, model-agreement status,
the saved comparison table, and the comparison chart. To deploy it with
Streamlit Community Cloud, push the repository and the four allowed model
artifacts to GitHub, then select `streamlit_app.py` as the app entry point.

The interface validates empty and oversized input, loads the two trusted model
artifacts once per application process, reports whether the models agree, and
handles missing or incompatible artifacts without exposing a debugger.

## Data Preparation

`prepare_data.py` performs the following operations:

1. Loads `data/messages.csv`.
2. Normalizes ham/spam labels.
3. Removes empty, invalid, conflicting, and duplicate messages.
4. Cleans punctuation, numbers, and repeated whitespace.
5. Creates a stratified 70/30 training and test split using random state 42.
6. Verifies that every training class supports five-fold cross-validation.
7. Checks that no cleaned message occurs in both training and test sets.
8. Saves the shared text and label splits to `data/processed/`.

## Feature Extraction and Model Selection

Both training scripts place TF-IDF feature extraction inside a scikit-learn
Pipeline. GridSearchCV compares unigram features with unigram + bigram features
using five-fold stratified cross-validation. Because TF-IDF is fitted separately
inside every fold, the validation fold cannot influence its vocabulary or IDF
weights.

Naive Bayes also searches smoothing strength and whether to learn class priors.
SVM uses LinearSVC and searches the regularization value and balanced/unbalanced
class weights. The best configuration is selected by spam-class F1-score, then
evaluated once on the untouched 30% test set.

## Model Evaluation

Both classifiers are evaluated on the same unseen test set using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The comparison step creates:

- `models/comparison_table.csv`
- `models/comparison_chart.png`
- `models/confusion_matrix_nb.png`
- `models/confusion_matrix_svm.png`
- `models/naive_bayes_grid_search.csv`
- `models/svm_grid_search.csv`
- `models/naive_bayes_training_summary.json`
- `models/svm_training_summary.json`

The CSV table and figures can be included in the Results and Discussion
sections of the assignment documentation.

The `data/processed/` directory and intermediate files in `models/` are ignored
by Git. The two trained pipelines, comparison table, and comparison chart are
allowed through `.gitignore` because the deployed web interface requires them.
After retraining, commit those four deployment artifacts with the application.

## Run the Tests

```bash
python -m unittest discover -s tests -v
```
