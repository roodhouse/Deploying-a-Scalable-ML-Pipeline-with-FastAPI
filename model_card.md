# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

I built this model for my D501 course project (Deploying a Scalable ML Pipeline with FastAPI). It’s a binary classifier that predicts whether someone earns more than $50K a year (`>50K`) or not (`<=50K`).

- **Who built it:** James Rugh  
- **When:** July 2026  
- **Version:** 1.0  
- **Model type:** Random Forest classifier (`sklearn.ensemble.RandomForestClassifier` with default settings)  
- **What it uses:** Census-style features from `data/census.csv` — things like age, education, occupation, hours worked, capital gain/loss, plus categorical fields such as workclass, marital status, relationship, race, sex, and native country. Categorical columns are one-hot encoded; the label is binarized so `>50K` is the positive class.  
- **How I trained it:** 80/20 train/test split (`random_state=42`). I fit the encoder and model on the training set only, then saved `model/model.pkl` and `model/encoder.pkl`.  
- **More reading:** Mitchell et al. model card paper (https://arxiv.org/pdf/1810.03993.pdf); Adult/Census Income data (UCI).  
- **Questions:** Repo issues at https://github.com/roodhouse/Deploying-a-Scalable-ML-Pipeline-with-FastAPI  

## Intended Use

I’m using this as a **learning project** — to practice training, evaluating slices, saving artifacts, and later serving predictions from an API.

- **Intended use:** Coursework and portfolio demos of a tabular income classifier pipeline.  
- **Intended users:** Me, instructors, and anyone reviewing this as student work.  
- **Not for:** Real decisions about loans, jobs, housing, benefits, or anything that affects a real person. This is not a production fairness-audited system.

## Training Data

I trained on 80% of `data/census.csv` after a random split (**26,048** rows, `test_size=0.20`, `random_state=42`).

The full file is skewed toward lower income: about **76%** `<=50K` and **24%** `>50K`. Categorical features are one-hot encoded with `OneHotEncoder(handle_unknown="ignore")` fit on train only. Continuous columns are used as-is (no scaling). After processing, the feature matrix has **108** columns.

Some categories are rare (certain countries, `Without-pay`, etc.), so the model doesn’t see many examples of those groups during training.

## Evaluation Data

I evaluated on the held-out **20%** of the same census file (**6,513** rows).

I used the same columns and the **training-fitted** encoder and label binarizer (`process_data` with `training=False`). That way the test set is processed the same way the model saw at train time, without refitting the encoder on test data.

Slice metrics in `slice_output.txt` are also computed on this test set, holding one categorical feature value fixed at a time.

## Metrics

I report **precision**, **recall**, and **F1** (beta = 1) for the positive class `>50K`, using scikit-learn with `zero_division=1`.

**Overall on the test set:**

| Metric | Score |
|---|---|
| Precision | **0.7387** |
| Recall | **0.6244** |
| F1 | **0.6768** |

So when the model predicts `>50K`, it’s right about **74%** of the time, but it only catches about **62%** of people who actually earn `>50K`. The F1 balances those two at about **0.68**.

I also computed the same three metrics on **categorical slices** (see `slice_output.txt`). A few highlights:

- **Sex:** Male F1 **0.6940** (recall 0.6517) vs Female F1 **0.5648** (recall 0.4678) — the model misses more high-income women.  
- **Race (larger groups):** White F1 **0.6755**, Black F1 **0.6372**, Asian-Pac-Islander F1 **0.7500**.  
- **Education:** Higher education tends to score better (e.g. Doctorate F1 **0.8718**, Masters **0.8317**); some lower-education slices are weak (e.g. 10th F1 **0.2222**, 7th-8th F1 **0.0000** on n = 141).

Very small slices can show perfect or zero scores that aren’t trustworthy because of sample size.

I did **not** tune a custom probability threshold; predictions use the Random Forest’s default class vote.

## Ethical Considerations

Income prediction can be harmful if someone used it for credit, hiring, or benefits. I’m only using this for class.

The data includes sensitive attributes (sex, race, country). My slice results already show uneven performance — especially lower recall for women. Historical bias in the census labels can get baked into the model.

I didn’t apply fairness constraints or rebalancing. Missing/unknown values (like workclass `?`) are treated like any other category, which may hide quality problems that hit some groups harder than others.

## Caveats and Recommendations

**Caveats:** Default Random Forest hyperparameters, no threshold tuning, no scaling of numeric features, class imbalance not fixed, and slice scores on rare categories can look extreme. I only looked at one feature at a time, not combinations like race × sex.

**What I’d do next / recommend:**

1. Keep this offline and educational unless a full fairness and legal review is done.  
2. Keep tracking precision, recall, and F1 by group (and add intersectional slices later).  
3. If fairness matters, try rebalancing, threshold adjustment, or fairness-aware methods.  
4. Be careful about using race/sex as features in any real application.  
5. Retrain and update this card whenever the data or pipeline changes, and bump the version.
