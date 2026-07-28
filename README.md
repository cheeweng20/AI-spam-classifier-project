# Message Spam Classifier - Supervised Machine Learning

This project classifies messages as ham or spam using two supervised
classification algorithms:

- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

Both models are trained and evaluated using the same cleaned data, feature
representation, training set, and test set so their results can be compared
fairly.

## Project Structure

```text
AI-spam-classifier-project/
|-- data/
|   |-- messages.csv
|   `-- processed/
|-- models/
|-- src/
|   |-- prepare_data.py
|   |-- train_naive_bayes.py
|   |-- train_svm.py
|   |-- compare_models.py
|   `-- app.py
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
python src/app.py
```

The web application will be available at:

```text
http://127.0.0.1:5000
```

## Data Preparation

`prepare_data.py` performs the following operations:

1. Loads `data/messages.csv`.
2. Normalizes ham/spam labels.
3. Removes empty, invalid, conflicting, and duplicate messages.
4. Cleans punctuation, numbers, and repeated whitespace.
5. Creates a stratified 70/30 training and test split using random state 42.
6. Fits a CountVectorizer on the training messages only.
7. Saves the shared training artifacts to `data/processed/`.

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

The CSV table and figures can be included in the Results and Discussion
sections of the assignment documentation.

The `data/processed/` and `models/` directories are generated locally and are
ignored by Git to prevent model and result-file conflicts between teammates.
After cloning the repository, run the complete workflow above to create them.

## Run the Tests

```bash
python -m unittest discover -s tests -v
```
