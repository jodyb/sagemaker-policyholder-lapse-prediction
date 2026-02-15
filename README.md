# Policyholder Lapse Prediction — SageMaker ML Pipeline

An end-to-end machine learning pipeline that predicts which motor vehicle insurance policyholders are likely to lapse (not renew), built on Amazon SageMaker.

## Business Problem

Insurance companies lose revenue when policyholders don't renew. Identifying at-risk customers before renewal allows targeted retention campaigns — personal agent calls for high-risk customers, email campaigns for medium-risk, and standard processing for low-risk.

This model scores policyholders into three risk tiers:

| Tier | Probability | Actual Lapse Rate | Recommended Action |
|------|------------|-------------------|-------------------|
| High | ≥ 0.60 | ~61% | Personal agent call + retention offer |
| Medium | 0.30 – 0.60 | ~19% | Targeted email + early renewal discount |
| Low | < 0.30 | ~3% | Standard renewal process |

## Architecture
```
Raw Data (S3)
    │
    ▼
Preprocessing (SageMaker Processing / SKLearn)
    │
    ▼
Training (SageMaker Training / XGBoost)
    │
    ▼
Model Registry (Versioned, Approval Workflow)
    │
    ▼
Batch Transform (Batch Scoring)
    │
    ▼
Risk Tiering (High / Medium / Low)
```

## Dataset

Motor vehicle insurance data: 105,555 policyholders × 26 features.

- **Target:** Lapsed (binary — 1 = did not renew, 0 = renewed)
- **Lapse rate:** ~20.4%
- **Split:** Train 60% (63,274) / Validation 20% (21,111) / Test 20% (21,170)

## Key Results

- **Algorithm:** XGBoost (200 trees)
- **Recall at 0.20 threshold:** 97% — catches nearly all lapsers
- **Risk tiers validated on holdout test set** — performance consistent across validation and test data
- **Top predictive features (SHAP):** Customer tenure, Seniority, Policies in force

## Project Structure
```
├── notebooks/
│   ├── 01-eda.ipynb                  # Exploratory Data Analysis
│   ├── 02-preprocessing.ipynb        # SageMaker Processing job
│   ├── 03-xgboost-training.ipynb     # Baseline XGBoost training
│   ├── 04-tuning-evaluation.ipynb    # Threshold analysis & hyperparameter tuning
│   ├── 05-feature-importance.ipynb   # Feature importance & SHAP explainability
│   ├── 06-deployment.ipynb           # Batch Transform, Model Registry, risk tiering
│   └── 07-final-evaluation.ipynb     # Holdout test set evaluation
├── src/
│   ├── data/
│   │   └── preprocess.py             # Preprocessing script (SageMaker Processing)
│   └── inference/
│       └── predict.py                # Production batch inference script
└── README.md
```

## Notebooks

Each notebook includes a markdown header explaining its objective, what it covers, and key findings.

1. **EDA** — Dataset exploration, class imbalance analysis, feature distributions
2. **Preprocessing** — Feature engineering, train/val/test split, SageMaker Processing job
3. **Training** — XGBoost baseline, endpoint deployment, confusion matrix evaluation
4. **Tuning** — Threshold optimization (0.20–0.70), Bayesian hyperparameter tuning (20 jobs)
5. **Feature Importance** — Gain-based importance, SHAP summary and waterfall plots
6. **Deployment** — Batch Transform, Model Registry with approval workflow, risk tiering
7. **Final Evaluation** — Holdout test set scoring, validation vs test comparison

## SageMaker Services Used

- **SageMaker Processing** — Managed preprocessing with SKLearnProcessor
- **SageMaker Training** — XGBoost model training with managed infrastructure
- **SageMaker Hyperparameter Tuning** — Bayesian optimization (20 jobs, 4 parallel)
- **SageMaker Endpoints** — Real-time inference for interactive evaluation
- **SageMaker Batch Transform** — Batch scoring with input_filter and join_source
- **SageMaker Model Registry** — Version tracking with approval workflow

## Production Inference

The `src/inference/predict.py` script provides a production-ready batch scoring pipeline:
```bash
python predict.py \
    --model-package-group insurance-lapse-prediction \
    --input-path s3://bucket/data/new-customers.csv \
    --output-path s3://bucket/predictions/ \
    --threshold 0.30
```

The script automatically retrieves the latest approved model from the Model Registry, runs Batch Transform, and outputs scored results with risk tiers.

## Author

Jody Baty
