# Analysis decisions

This record explains the choices behind the project, including their limitations. It distinguishes the original experiment from the later credibility review; it is not a claim that all choices were specified before the dataset was explored.

## 1. Frame the task as review at transaction initiation

The business question is whether an initiated TRANSFER or CASH_OUT warrants review or additional authentication. All fraud labels in this file occur in those types, so they define the modelling population. That observation does not establish that other payment types are safe in a real institution.

An alert is not proof of fraud or an automatic decline. The project reports transaction-level review volume, false positives and missed fraud. It does not invent investigation costs, recovered losses or deployment outcomes. Amounts remain in dataset currency units.

## 2. Challenge feature provenance before trusting the score

The original feature set used requested amount, type, recorded opening origin balance and relationships between amount and balance. Closing balances, residuals, account identifiers, the existing flag and the label were excluded from the model input.

The later source review found the [publisher's warning](https://www.kaggle.com/datasets/ealaxi/paysim1/data) about cancellation behaviour affecting balance fields. A column named “opening balance” is not sufficient evidence of an outcome-independent, initiation-time snapshot.

The original policy is therefore retained as a **historical experiment and diagnostic reference**. Amount/type-only models are the **conservative research baselines**, because they remove the disputed balance dependency. Neither is presented as ready for a bank.

Ablation makes the issue measurable. Logistic validation AP falls from 1.0000 to 0.6728 without the full-balance flag, to 0.0186 without explicit relationship features, and to 0.0033 without any balance information. Histogram boosting can infer some relationships without explicit ratios; removing a feature is not always equivalent to removing its information. These comparisons show dependence, not the exact causal mechanism of a simulator artefact.

## 3. Preserve chronological boundaries and the test's actual history

A random split would mix earlier and later activity and could conceal changes in transaction composition. The original whole-step split remains fixed: training at steps 1–323, validation at 324–377 and historical testing at 378–743. Preprocessing learns from training rows only.

The test results have already been inspected, and EDA used the full file. Repartitioning those same records would not create genuinely unseen evidence. The credibility review therefore keeps every new modelling experiment within steps ≤377. The later window is used only to reproduce the fixed policy and replay monitoring.

Three rolling development checks use separate fit, threshold-selection and forward-evaluation windows. They are expanding and correlated, not independent external tests. They also assume labels are known at the fitting cutoff: confirmation timestamps are absent, so a realistic label-delay gap cannot be established.

## 4. Separate ranking quality from the operating decision

Fraud is rare: accuracy would reward a model for ignoring it. AP summarises ranking performance, while precision, recall and F1 describe behaviour at an alert threshold. AP is interpreted alongside the fraud prevalence of the same population.

The original model was chosen by validation AP; an exact tie favoured the first, interpretable Logistic Regression candidate. Its threshold maximised validation F1, with precision and then the higher threshold breaking ties. The fitted pipeline was not refitted on train plus validation, which could change the score distribution.

F1 is a transparent research convention, not an economic optimum. In one rolling window, the amount/type-only logistic model alerts on 19.95% of transactions after its threshold was selected in an earlier window. That result illustrates why a bank would need an agreed review budget, customer-friction limits and loss definitions before selecting a policy. The project does not tune the threshold against this failure or the historical test.

## 5. Explain a changing rate through its numerator and denominator

Fraud prevalence rises from 0.1364% in validation to 0.9827% in the historical test. Fraud counts per elapsed step are similar, while legitimate counts per step fall substantially. The final 296 records are all labelled fraud, but the earlier part of the historical test still has a much higher prevalence than validation. The small transaction-type mix change also fails to account for the whole gap.

These are observations from the file. Simulation scheduling, fraud injection and extraction coverage are possible explanations, not verified causes. The project does not claim that an actual bank experienced this risk trend.

## 6. Monitor only what is available, and acknowledge weak rules

Volume, amount, type mix, scores and alert counts can be monitored without confirmed fraud outcomes, subject to the balance-dependent policy's provenance caveat. Precision, recall, false positives and missed fraud require labels. The historical prototype uses retrospectively supplied labels; it cannot reconstruct contemporaneous label maturity.

Reference bands and numeric bins come from development data. PSI and type-share watch levels are explicitly illustrative. The 54-step reference already contains extreme per-step alert rates, so broad bands can miss meaningful changes. The correct response is to document that limitation and seek a longer development reference, not tighten limits after viewing the replay.

Changes should trigger checks of data coverage, feature mapping, denominator changes and label maturity before a model revision. No automatic retraining, blocking or production alert delivery is claimed.

## Evidence and next decision

The complete [results report](RESULTS.md), [rolling-window definitions](../reports/tables/development_rolling_windows.csv) and [verification record](../reports/verification.json) support these decisions. The next substantive step is verified as-of data and genuinely new future evidence, followed by a business-approved investigation objective. More complex models alone do not address the current limitations.
