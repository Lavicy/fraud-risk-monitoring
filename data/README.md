# Local dataset

Obtain **Synthetic Financial Datasets For Fraud Detection** from the [publisher's Kaggle dataset page](https://www.kaggle.com/datasets/ealaxi/paysim1/data), following the site's access and licence terms. Extract the CSV and place it here as:

```text
data/PS_20174392719_1491204439457_log.csv
```

The [PaySim simulator repository](https://github.com/EdgarLopezPhD/PaySim) links to this dataset. Its associated paper is E. A. Lopez-Rojas, A. Elmir and S. Axelsson, *PaySim: A financial mobile money simulator for fraud detection*, EMSS, 2016.

This is synthetic mobile-money data, not a UK bank's customer records. The publisher warns that cancellation behaviour affects the balance columns, including opening balances. See Notebook 02 for the implications. Amounts are labelled **dataset currency units** throughout this project; no GBP conversion is assumed.

The inspected local CSV contains 6,362,620 rows. Its SHA-256 is:

```text
16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b
```

The fingerprint identifies the file analysed; it does not establish its original download history or independently authenticate the contents. Notebook 01 checks the five columns it reads and saves `reports/tables/data_quality.csv`. The main model uses only amount and type. The earlier balance-dependent specification remains in [Git history](https://github.com/Lavicy/fraud-risk-monitoring/tree/c4519cb66e21e7f3aa88f990cfd301e423b0a276).

The existing CSV is preserved locally. `.gitignore` excludes raw files in this directory; only this note is intended for version control. No download credentials are needed by the notebooks once the file is present.
