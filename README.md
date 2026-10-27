# Heart Failure Risk Explorer
Educational ML project by Stephanie Lee. **Not for clinical use.**

## What it does
Predicts risk of death during follow-up for heart-failure patients from routine clinical measurements, and serves the model through a small Gradio web app.

## Data
UCI *Heart Failure Clinical Records* (299 patients, 12 features), CC BY 4.0. Chicco & Jurman (2020), BMC Med Inform Decis Mak 20:16.

## Method
- Stratified 80/20 train/test split; 5-fold cross-validation for model comparison.
- Follow-up `time` excluded (it leaks the outcome: it is not known at prediction time).
- Model: Random Forest.

## Results (held-out test set, threshold 0.5)
| Metric | Value |
|---|---|
| ROC-AUC (test) | 0.79 |
| ROC-AUC (5-fold CV mean) | 0.79 |
| Recall (deaths caught) | 0.53 |
| Precision | 0.59 |

## Limitations
- Only 299 patients from a single region and time period; results may not generalize.
- Small test set, so metrics are noisy.
- Association, not causation. Never use for real medical decisions.

## Run it
`pip install -r requirements.txt && python app.py`
