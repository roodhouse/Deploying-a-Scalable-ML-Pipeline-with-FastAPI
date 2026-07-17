# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Developer:** James Rugh (roodhouse) / Western Governors University D501 project work.
- **Model date:** July 2026.
- **Model version:** 1.0.0.
- **Model type:** Binary classifier that predicts whether an individual’s annual income is greater than $50,000 (`>50K`) or at most $50,000 (`<=50K`).
- **Algorithm and features:** A scikit-learn `RandomForestClassifier` trained with default hyperparameters (no explicit fairness constraints applied at training time). Continuous inputs include age, education-num, capital-gain, capital-loss, hours-per-week, and fnlgt. Categorical inputs are one-hot encoded and include workclass, education, marital-status, occupation, relationship, race, sex, and native-country. Labels are binarized with `LabelBinarizer` (`>50K` as the positive class).
- **Training approach:** An 80/20 train–test split with `random_state=42`. Categorical features are encoded with `OneHotEncoder(handle_unknown="ignore")` fit only on the training fold. The trained model and encoder are serialized to `model/model.pkl` and `model/encoder.pkl`.
- **Paper / resources:** Model Card paper (Mitchell et al.): https://arxiv.org/pdf/1810.03993.pdf. Census Income (Adult) dataset documentation: https://archive.ics.uci.edu/ml/datasets/census+income.
- **Citation:** Becker, B. & Kohavi, R. Census Income Data Set. UCI Machine Learning Repository.
- **License:** See `LICENSE.txt` in this repository for project licensing. The underlying Adult/Census Income data is from the UCI Machine Learning Repository.
- **Questions or comments:** Open an issue on the project repository: https://github.com/roodhouse/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.

## Intended Use

- **Primary intended uses:** Educational demonstration of an end-to-end ML pipeline (data processing, training, slice-based evaluation, model persistence, and later API serving). Suitable for offline research-style analysis of income prediction on the Adult census features.
- **Primary intended users:** Students, instructors, and practitioners reviewing a teaching portfolio or learning deployment of a simple tabular classifier.
- **Out-of-scope use cases:** Real credit, lending, hiring, insurance, or government eligibility decisions; any high-stakes screening of individuals; production use without additional bias auditing, monitoring, and legal review. This model is **not** a substitute for a production fair-lending or employment system.

## Factors

- **Relevant factors:** Demographic attributes present in the data (sex, race, native-country), socioeconomic proxies (education, occupation, workclass, hours-per-week, capital gain/loss), and family/relationship status. Many of these are sensitive or highly correlated with protected characteristics.
- **Evaluation factors:** Overall test-set precision, recall, and F1, plus the same metrics computed on slices where a single categorical feature is held fixed (workclass, education, marital-status, occupation, relationship, race, sex, native-country), as reported in `slice_output.txt`.

## Metrics

Metrics are computed with scikit-learn on the held-out test set using the positive class `>50K`, with `zero_division=1` for edge cases.

| Metric | Definition | Overall test performance |
|---|---|---|
| **Precision** | Of predicted `>50K`, share that are truly `>50K` | **0.7387** |
| **Recall** | Of true `>50K`, share correctly predicted | **0.6244** |
| **F1** | Harmonic mean of precision and recall (β = 1) | **0.6768** |

**Decision threshold:** The Random Forest’s default class prediction rule (argmax of class probabilities / majority of trees); no custom probability threshold was tuned.

**Variation approaches:** Performance is also reported per categorical slice in `slice_output.txt` (unitary results by feature value). Intersectional slices (e.g., race × sex) were not computed in this run.

### Example slice results (from `slice_output.txt`)

**Sex**

| Slice | Count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Female | 2,126 | 0.7124 | 0.4678 | 0.5648 |
| Male | 4,387 | 0.7421 | 0.6517 | 0.6940 |

**Race** (selected)

| Slice | Count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| White | 5,595 | 0.7364 | 0.6239 | 0.6755 |
| Black | 599 | 0.7500 | 0.5538 | 0.6372 |
| Asian-Pac-Islander | 193 | 0.7759 | 0.7258 | 0.7500 |
| Amer-Indian-Eskimo | 71 | 0.6250 | 0.5000 | 0.5556 |
| Other | 55 | 1.0000 | 0.6667 | 0.8000 |

**Education** (illustrative high/low F1, n ≥ 50)

| Slice | Count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Doctorate | 77 | 0.8500 | 0.8947 | 0.8718 |
| Masters | 369 | 0.8278 | 0.8357 | 0.8317 |
| Bachelors | 1,053 | 0.7500 | 0.7000 | 0.7241 |
| HS-grad | 2,085 | 0.6364 | 0.4261 | 0.5104 |
| 10th | 183 | 0.3333 | 0.1667 | 0.2222 |
| 7th-8th | 141 | 1.0000 | 0.0000 | 0.0000 |

Small slices (low count) can show extreme metrics (0.0 or 1.0) that are unstable and should not be over-interpreted.

## Evaluation Data

- **Datasets:** Held-out **20%** of `data/census.csv` after an 80/20 train–test split (`test_size=0.20`, `random_state=42`), yielding **6,513** evaluation rows.
- **Motivation:** Estimate generalization of income classification on Adult/Census Income–style records for teaching and pipeline validation, including group-wise slice metrics.
- **Preprocessing:** Same feature columns as training; categorical columns one-hot encoded with the **training-fitted** encoder (`training=False`); label binarized with the training-fitted `LabelBinarizer`. Rows are not rebalanced at evaluation time.

## Training Data

- **Datasets:** **80%** of `data/census.csv` (**26,048** rows) from the same split as evaluation.
- **Class balance (full census file):** approximately **75.9%** `<=50K` and **24.1%** `>50K` (imbalanced toward lower income).
- **Motivation:** Provide enough labeled tabular examples to train a Random Forest on mixed continuous and categorical census features.
- **Preprocessing:** Categorical features listed in `train_model.py` are one-hot encoded; continuous features are passed through without scaling; label is `salary`. Feature matrix width after processing is **108** columns in this run.
- **Distribution notes:** Some categorical values (e.g., rare native-country codes, `Without-pay` workclass) appear infrequently; slice metrics for those groups are high-variance.

## Quantitative Analyses

### Unitary results

- Overall test metrics: **Precision 0.7387**, **Recall 0.6244**, **F1 0.6768**.
- By sex, F1 is higher for **Male (0.6940)** than **Female (0.5648)**, driven largely by lower recall for women (0.4678 vs 0.6517).
- By race, F1 ranges from about **0.56** (Amer-Indian-Eskimo) to **0.75** (Asian-Pac-Islander) among larger groups; White F1 is **0.6755**, close to overall performance.
- By education, advanced degrees (Doctorate, Prof-school, Masters) achieve F1 above **0.83**, while some lower-education slices show very low recall (e.g., 7th-8th F1 **0.0000** on n = 141).

### Intersectional results

Intersectional analyses (e.g., race × sex, education × sex) were **not** computed in the current pipeline. Only single-feature (unitary) slices were written to `slice_output.txt`. A future iteration should add intersectional reporting before any real-world use.

## Ethical Considerations

- Income prediction can affect access to credit, housing, and opportunity if misused. This model is for **education only**.
- Protected or sensitive attributes (sex, race, native-country) are used as features, which can encode historical bias. Slice metrics already show **disparate performance** (e.g., lower recall/F1 for Female vs Male).
- Missing or placeholder values (e.g., workclass `?`) appear in the data; treating them as ordinary categories can hide data-quality issues that fall unevenly across groups.
- High metrics on tiny slices can create a false sense of fairness; sample sizes must be reported alongside scores (as in `slice_output.txt`).
- No fairness constraints, reweighting, or equalized-odds post-processing were applied.

## Caveats and Recommendations

- **Caveats:** Default Random Forest hyperparameters; no threshold tuning; no continuous feature scaling; class imbalance is unaddressed; slice metrics for rare categories are unreliable; intersectional performance is unknown.
- **Recommendations:**
  1. Do not deploy for consequential decisions without legal, ethical, and fairness review.
  2. Monitor group-wise precision, recall, and F1 (and add intersectional slices) on a schedule.
  3. Consider rebalancing, threshold selection by group, or fairness-aware methods if equity goals matter.
  4. Prefer removing or carefully handling sensitive attributes and proxies when the use case does not require them.
  5. Re-train and re-document when the data distribution or feature set changes; bump the model version and update this card with new overall and slice metrics.
