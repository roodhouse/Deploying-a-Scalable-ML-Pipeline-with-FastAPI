# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random Forest classifier built by John Rugh. The model predicts whether annual income is `>50K` or `<=50K` from Census features (age, education, occupation, hours worked, capital gain/loss, workclass, marital status, relationship, race, sex, native country, and related fields). Categorical features are one-hot encoded and no fairness constraints were applied during training.

## Intended Use

For coursework and portfolio demonstration of an ML training and evaluation pipeline. Intended users are students and instructors. Not intended for real credit, hiring, lending, or other decisions about people.

## Training Data

Trained on 80% of `data/census.csv` (26,048 rows; `train_test_split` with `test_size=0.20`, `random_state=42`). The full census file is about 76% `<=50K` and 24% `>50K`. Categorical columns are one-hot encoded with an encoder fit on the training set only.

Census Income data from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data

Evaluated on the held-out 20% of the same file (6,513 rows), processed with the training-fitted encoder and label binarizer. Slice metrics in `slice_output.txt` use this same test set, fixing one categorical feature value at a time.

## Metrics

_Please include the metrics used and your model's performance on those metrics._

Precision, recall, and F1 (positive class `>50K`) on the test set:

- **Precision: 0.7387**
- **Recall: 0.6244**
- **F1: 0.6768**

When analyzing across data slices, performance varies by group. For example, F1 is higher for males (0.6940) than females (0.5648), and higher for advanced education levels (e.g. Doctorate F1 0.8718) than some lower education slices (e.g. 10th F1 0.2222). Small slices can show extreme scores and should be read with care.

## Ethical Considerations

Income prediction can affect real opportunities if misused; this model is for class only. The model is not intended to inform decisions about matters central to human life or flourishing.

## Caveats and Recommendations

**Did the results suggest further testing?** Yes. Overall F1 is about 0.68, but slices show clear gaps (for example lower recall/F1 for females than males, and weak performance on some education levels). That points to more testing: intersectional slices (e.g. race × sex), threshold tuning, and checks on groups with low counts before any broader use.

**Were any relevant groups missing from evaluation?** Mostly no for major groups—sex, race, and common education/workclass values appear in the test set. A few categories that appear in training are missing or nearly missing from the test split with this `random_state` (e.g. workclass `Never-worked`, native-country `Holand-Netherlands` and `Outlying-US(Guam-USVI-etc)`). Many other native-country values have very few test rows (under 10), so they are underrepresented for reliable metrics even when present.

**Additional recommendations for use:** Keep this model offline and educational. Do not use it for credit, hiring, or benefits. If the pipeline is extended, monitor group-wise precision/recall/F1, be careful with sensitive features, and retrain when the data changes.

**Ideal evaluation data for this model:** A held-out set drawn from the same population and feature schema as training, large enough that important demographic and socioeconomic slices have adequate sample size, with label balance closer to the real deployment mix (or explicitly reweighted). It should include the same categorical values seen in training (or clearly document new levels), allow intersectional analysis, and stay free of leakage from the training fold.

