# Customer Churn Prediction ML Pipeline

[![CI](https://github.com/SourabhaKK/customer-churn-prediction-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/SourabhaKK/customer-churn-prediction-ml/actions/workflows/ci.yml)

End-to-end scikit-learn ML pipeline for customer churn prediction — leakage-safe preprocessing, TDD methodology, and joblib model export for downstream serving.

---

## Architecture

```mermaid
graph TD
    subgraph DATA["DATA LAYER"]
        CSV["Raw CSV<br/>data/customer_churn_dataset.csv"]
        DV["Data Validator<br/>src/data_validation.py"]
        FP["Feature Preparation<br/>_prepare_features: TotalCharges coerce · customerID drop"]
        CSV --> DV
        DV -->|"validates schema"| FP
    end

    subgraph PREP["PREPROCESSING"]
        SPLIT["Train/Test Split<br/>80/20 stratified · random_state=42"]
        FIT["Preprocessing Fit<br/>fit_preprocess: StandardScaler + OHE fit on train only"]
        TRANS["Preprocessing Transform<br/>transform_preprocess on test"]
        SPLIT -->|"fit on train only"| FIT
        SPLIT -->|"transform only"| TRANS
    end

    subgraph ML["ML LAYER"]
        FE["Feature Engineering<br/>src/features.py: tenure groups · charges ratio"]
        TRAIN["Model Training<br/>RandomForestClassifier · 100 estimators · random_state=42"]
        EVAL["Model Evaluation<br/>ROC-AUC · Accuracy · F1 on test set only"]
        FE -->|"51 features"| TRAIN
        TRAIN --> EVAL
    end

    subgraph EXP["EXPORT"]
        EXPSCRIPT["Export Script<br/>scripts/export_model.py"]
        ARTIFACT["models/churn_model.joblib"]
        EXPSCRIPT --> ARTIFACT
    end

    subgraph DOWN["DOWNSTREAM"]
        FASTAPI["ml-model-deployment-fastapi"]
    end

    FP --> SPLIT
    FIT --> FE
    TRANS --> FE
    EVAL -->|"ROC-AUC 0.839"| EXPSCRIPT
    ARTIFACT -->|"joblib.load()"| FASTAPI
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant CSV
    participant DataValidator
    participant FeatureEngineer
    participant Preprocessor
    participant Trainer
    participant Evaluator
    participant Exporter

    CSV->>DataValidator: 7,043 rows · 21 columns
    DataValidator->>DataValidator: validate schema · coerce 11 blank TotalCharges to 0.0
    DataValidator->>FeatureEngineer: stratified 80/20 train/test splits
    FeatureEngineer->>FeatureEngineer: add tenure_group · avg_monthly_charge · charge_ratio
    FeatureEngineer->>Preprocessor: train partition (5,634 rows · 22 columns)
    Preprocessor->>Preprocessor: fit StandardScaler + OHE on train only
    Preprocessor->>Trainer: X_train (5,634 × 51)
    FeatureEngineer->>Preprocessor: test partition (1,409 rows · 22 columns)
    Preprocessor->>Evaluator: X_test (1,409 × 51) — transform only, no refit
    Trainer->>Trainer: RandomForestClassifier(n_estimators=100, random_state=42).fit()
    Trainer->>Evaluator: fitted model
    Evaluator->>Evaluator: ROC-AUC · Accuracy · F1 on test set only
    Evaluator->>Exporter: roc_auc=0.839 · accuracy=0.7991
    Exporter->>Exporter: joblib.dump(artifact_dict, compress=3)
    Exporter-->>Exporter: models/churn_model.joblib written
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Dataset | 7,043 rows · 19 raw features |
| Train / Test split | 5,634 / 1,409 (stratified 80/20) |
| Post-encoding features | 51 |
| Model | RandomForestClassifier (100 estimators) |
| ROC-AUC | 0.839 |
| Accuracy | 79.91% |
| Churn-class F1 | +0.58 vs majority baseline |
| Test suite | 92 tests · 100% pass rate |
| Inference latency | 16.8 ms (single-row) |

---

## Engineering Highlights

- **Leakage-safe preprocessing**: `StandardScaler` and `OneHotEncoder` are fit exclusively on the training partition via `fit_preprocess`, then applied to test data via `transform_preprocess`. A dedicated leakage test class enforces no refitting on test data.
- **TotalCharges blank-string handling**: 11 rows in the raw Telco CSV store `TotalCharges` as blank strings. The pipeline coerces them via `pd.to_numeric(errors='coerce').fillna(0)` with a `WARNING` log before validation fires.
- **customerID dropped before encoding**: `_prepare_features` drops `customerID` prior to `ColumnTransformer` to prevent 7,043 spurious OHE columns from inflating the feature space.
- **Deterministic evaluation**: `random_state=42` is fixed at every call site — `train_test_split`, `RandomForestClassifier`, and `train_model` — guaranteeing bit-for-bit reproducible metrics across runs.
- **Export artifact contains model + metadata**: `scripts/export_model.py` serialises a dict with `model`, `feature_count`, `model_version`, `roc_auc`, `accuracy`, and `trained_on` timestamp. The serving layer validates `feature_count` on load to catch silent shape mismatches.
- **9 TDD cycles across 6 src modules**: every function was written test-first; RED→GREEN→REFACTOR screenshots are preserved in `outputs/`.

---

## Project Structure

```
customer-churn-prediction-ml/
├── src/
│   ├── data_validation.py          # Schema and data quality checks
│   ├── preprocessing.py            # Leakage-safe fit/transform pipeline
│   ├── features.py                 # Derived feature engineering
│   ├── train.py                    # RandomForestClassifier training
│   ├── evaluate.py                 # Metrics on held-out test set
│   ├── pipeline.py                 # End-to-end orchestration
│   └── predict.py                  # Inference interface
├── scripts/
│   └── export_model.py             # Trains and exports churn_model.joblib
├── tests/                          # 92 pytest cases across 8 modules
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_training.py
│   ├── test_prediction.py
│   ├── test_evaluate.py
│   ├── test_export.py
│   └── test_integration_regression.py
├── models/
│   └── README.md                   # Artifact docs (joblib excluded from git)
├── data/
│   └── customer_churn_dataset.csv
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/SourabhaKK/customer-churn-prediction-ml
cd customer-churn-prediction-ml
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python -m src.pipeline
```

Export the trained model:

```bash
python scripts/export_model.py
```

Run tests:

```bash
pytest tests/ -v
```

---

## CI/CD

The CI pipeline triggers on every push and pull request to `main`. It runs four steps in order on `ubuntu-latest` with Python 3.11: repository checkout (`actions/checkout@v4`), Python environment setup (`actions/setup-python@v5`), dependency installation (`pip install -r requirements.txt`), and the full test suite (`pytest tests/ -v --tb=short`). The badge above reflects the current status of the `ci.yml` workflow on the `main` branch.

---

## Ecosystem Position

This pipeline is the training layer of a connected ML system:

| Layer | Repo | Role |
|-------|------|------|
| ML Training | customer-churn-prediction-ml | ← YOU ARE HERE |
| LLM Backend | llm-ai-basket-builder | GPT-4o-mini + Pydantic + FastAPI |
| Model Serving | ml-model-deployment-fastapi | Serves churn model · Docker · AWS EC2 |
| Drift Detection | ml-model-monitoring-drift-detection | PSI / KS / Chi-Square · CLI |
| NLP Pipeline | nlp-complaint-classification-pipeline | TF-IDF + BERT · 253 tests |

---

## Engineering Notes

- **Why fit/transform separation instead of `fit_transform` on the full dataset**: calling `fit_transform` on the full dataset before splitting allows test-set statistics (mean, variance, category frequencies) to leak into the scaler and encoder, producing optimistically biased evaluation metrics. `fit_preprocess` + `transform_preprocess` prevents this by ensuring the transformer sees only training data during fitting.

- **Why `TotalCharges` required explicit coercion**: the Telco CSV stores `TotalCharges` as `dtype=object` because 11 rows contain blank strings for customers with zero tenure. Without `pd.to_numeric(errors='coerce').fillna(0)`, the column lands in the categorical pipeline (OHE) instead of the numeric pipeline (StandardScaler), producing incorrect feature types and silent value corruption.

- **Why `customerID` must be dropped before OHE**: `customerID` is a unique string identifier. If it reaches `OneHotEncoder`, it produces one binary column per customer (7,043 columns), inflating memory use, training time, and model complexity without adding predictive signal. `_prepare_features` removes it before any `ColumnTransformer` step.

- **Why `random_state=42` at every split point**: a single fixed seed is not sufficient if `train_test_split`, `RandomForestClassifier`, and `train_model` each use independent sources of randomness. Fixing all three guarantees that re-running the pipeline on the same dataset always produces the same split, the same tree structure, and the same evaluation metrics — making results auditable and reproducible.

- **Why the export artifact stores `feature_count` alongside the model**: the FastAPI serving layer calls `joblib.load()` and immediately checks `artifact["feature_count"] == 51` against the shape of incoming request data. Without this guard, a feature engineering change that adds or removes a column would cause a silent shape mismatch at inference time rather than a loud failure at export time. Storing the count in the artifact makes the contract between training and serving explicit and machine-checkable.
