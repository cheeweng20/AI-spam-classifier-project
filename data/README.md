# Dataset provenance

`messages.csv` is a consolidated CSV representation of the Ling-Spam corpus
introduced by Androutsopoulos et al. (2000). The corpus contains 2,893 English
email messages: 2,412 legitimate (ham) emails and 481 spam emails.

Primary source:

- Androutsopoulos, I., Koutsias, J., Chandrinos, K. V., Paliouras, G., &
  Spyropoulos, C. D. (2000). *An evaluation of Naive Bayesian anti-spam
  filtering*. https://arxiv.org/abs/cs/0006013

The local CSV has two required columns:

- `message`: email content
- `label`: `ham` or `spam`

Integrity information for the submitted CSV:

```text
Rows: 2,893
SHA-256: A26547C711F28EB16B3CB9DC8462D43750706330E4A25EB6D9B93BC0126CF71A
```

`src/prepare_data.py` validates the schema and labels, cleans the text, rejects
conflicting labels, removes normalized duplicates, creates a stratified 70/30
split, and verifies that no cleaned message occurs in both splits.
