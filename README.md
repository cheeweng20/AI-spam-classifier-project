# SMS Spam Classifier — AI Assignment (Supervised Machine Learning)

## Project Structure
```
spam-classifier-project/
├── data/
│   └── SMSSpamCollection      ← Download and place this file here (see Step 1)
├── src/
│   ├── prepare_data.py        ← Run once: data cleaning, splitting, and vectorization (shared by both team members)
│   ├── train_naive_bayes.py   ← Your task
│   ├── train_svm.py           ← Teammate's task
│   ├── compare_models.py      ← Combine both results and generate the comparison table
│   └── app.py                 ← Flask web interface (for demonstration)
├── models/                    ← Generated automatically after training (model files, charts, comparison table)
└── requirements.txt
```

## Step 1: Download the Dataset

1. Open: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Click **"Download"** and extract the ZIP file.
3. Place the extracted `SMSSpamCollection` file into the `data/` folder.

## Step 2: Install the Dependencies

Open a terminal in the project root directory and run:

```bash
python -m pip install -r requirements.txt
```

Make sure you use the **same Python interpreter** for both installing packages and running the scripts.

To verify:

```bash
python -c "import sys; print(sys.executable)"
python -m pip show joblib Flask
```

If Windows PowerShell cannot find the correct Python installation but Anaconda is installed, either:

- Open **Anaconda Prompt** and run the commands above, or
- Use the full Python path in PowerShell:

```powershell
& "$HOME\anaconda3\python.exe" -m pip install -r requirements.txt
```

## Step 3: Run the Workflow (in Order)

```bash
# Select the dataset:
# enron -> reads data/enron_spam_data.csv
# sms    -> reads data/SMSSpamCollection

# 1. Prepare the Enron dataset
python src/prepare_data.py --dataset enron

# 2. Train the Naive Bayes model (your task)
python src/train_naive_bayes.py --dataset enron

# 3. Train the SVM model (teammate's task)
python src/train_svm.py --dataset enron

# 4. Compare both models (run only after both models have been trained)
python src/compare_models.py --dataset enron

# 5. Launch the web demo
python src/app.py
```

To train the SMS dataset instead, simply replace every `--dataset enron` with `--dataset sms`.

## Output Files

After the workflow finishes, the following files will be generated:

- `models/<dataset>/naive_bayes_model.joblib`
- `models/<dataset>/svm_model.joblib`

  Trained machine learning models.

- `models/<dataset>/confusion_matrix_nb.png`
- `models/<dataset>/confusion_matrix_svm.png`

  Confusion matrix visualizations for each model.

- `models/<dataset>/comparison_table.csv`

  A comparison table containing **Accuracy, Precision, Recall, and F1-score** for both models.
  **Insert this table directly into the Results section of your documentation/report.**

- `models/<dataset>/comparison_chart.png`

  A bar chart comparing both models.
  **Include this figure in your report as well.**

- Running:

```bash
python src/app.py
```

starts the Flask web application at:

```
http://127.0.0.1:5000
```

This web interface is intended for the project demonstration.

## Notes

- **Do not modify** `RANDOM_STATE = 42` in `prepare_data.py`.
  Both team members **must use the same random seed** to ensure a fair comparison.

- If you would like to compare **TF-IDF** with the current **CountVectorizer** implementation for extra credit, replace `CountVectorizer` with `TfidfVectorizer` (their usage in `scikit-learn` is almost identical). You may save this version as a separate file, such as `prepare_data_tfidf.py`.