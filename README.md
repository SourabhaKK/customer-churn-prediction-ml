# Customer Churn Prediction ML Pipeline

A production-quality machine learning project for predicting customer churn, built using **strict Test-Driven Development (TDD)** methodology with comprehensive test coverage and data leakage prevention.

## 📋 Project Overview

### Problem Statement
Customer churn (customer attrition) is a critical business metric where customers stop doing business with a company. Predicting which customers are likely to churn enables businesses to:
- Implement targeted retention strategies
- Reduce customer acquisition costs
- Improve customer lifetime value
- Optimize marketing spend

### Business Relevance
In subscription-based industries (telecom, SaaS, streaming services), acquiring a new customer costs 5-25x more than retaining an existing one. Accurate churn prediction allows proactive intervention before customers leave.

### Approach
This project implements an end-to-end ML pipeline with:
- **Strict TDD workflow** (79 comprehensive tests)
- **Data leakage prevention** (proper train/test separation)
- **Production-ready code** (type hints, error handling, documentation)
- **Reproducible results** (fixed random seeds)
- **Modular architecture** (separation of concerns)


---

## 🏗️ Project Structure

```
customer-churn-prediction-ml/
├── data/
│   └── customer_churn_dataset.csv          # Raw customer data
├── src/
│   ├── __init__.py
│   ├── data_validation.py                  # Schema & quality validation
│   ├── preprocessing.py                    # Feature scaling & encoding (leakage-safe)
│   ├── features.py                         # Feature engineering (tenure groups, ratios)
│   ├── train.py                            # Model training (RandomForest)
│   ├── predict.py                          # Prediction interface
│   ├── evaluate.py                         # Model evaluation & feature importance
│   └── pipeline.py                         # End-to-end orchestration
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py             # 8 tests
│   ├── test_preprocessing.py               # 18 tests (10 original + 8 leakage-safe)
│   ├── test_features.py                    # 12 tests
│   ├── test_training.py                    # 12 tests
│   ├── test_prediction.py                  # 14 tests
│   └── test_evaluate.py                    # 15 tests
├── outputs/
│   └── tdd_cycle*_*.png                    # TDD workflow screenshots (RED/GREEN phases)
├── example_feature_importance.py           # Feature importance example
├── requirements.txt                        # Python dependencies
├── pytest.ini                              # Pytest configuration
├── .gitignore                              # Git ignore rules
└── README.md                               # This file
```

### Module Descriptions


| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `data_validation.py` | Validates raw data schema and quality | `validate_dataframe(df)` |
| `preprocessing.py` | Scales numerical features, encodes categorical features | `fit_preprocess(df)`, `transform_preprocess(df, preprocessor)` |
| `features.py` | Creates derived features (tenure groups, charge ratios) | `engineer_features(df)` |
| `train.py` | Trains RandomForest classifier | `train_model(X, y, random_state)` |
| `predict.py` | Makes predictions with probability scores | `predict(model, X)` |
| `evaluate.py` | Computes evaluation metrics and feature importance | `evaluate_model(model, X_test, y_test)`, `get_feature_importance(model)` |
| `pipeline.py` | Orchestrates full workflow from raw data to evaluation | `run_pipeline(data_path)` |


---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/SourabhaKK/customer-churn-prediction-ml.git
cd customer-churn-prediction-ml
```

2. **Create virtual environment** (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `scikit-learn>=1.3.0` - ML models and preprocessing
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Test coverage

---

## 🧪 Usage

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run with verbose output:**
```bash
pytest -v
```

**Run specific test module:**
```bash
pytest tests/test_preprocessing.py -v
```

**Run with coverage report:**
```bash
pytest --cov=src --cov-report=html
```

**Expected output:** 64 tests passing ✅

### Running the End-to-End Pipeline

```python
from src.pipeline import run_pipeline

# Execute full pipeline
model, metrics = run_pipeline('data/customer_churn_dataset.csv')

# View evaluation metrics
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

**Pipeline steps:**
1. Load raw CSV data
2. Validate data schema and quality
3. Split into train/test (80/20, stratified)
4. Engineer features on both sets
5. Fit preprocessing on train set
6. Transform test set using train statistics
7. Train RandomForest model
8. Evaluate on test set
9. Return model and metrics

### Making Predictions on New Data

```python
from src.predict import predict
import pandas as pd

# Load new customer data
new_customers = pd.DataFrame({
    'tenure': [12, 36],
    'MonthlyCharges': [50.5, 75.2],
    'TotalCharges': [606.0, 2707.2],
    'Contract': ['Month-to-month', 'Two year']
})

# Make predictions
predictions = predict(model, new_customers)
print(predictions['predictions'])      # Class labels (0 or 1)
print(predictions['probabilities'])    # Probability scores
```

---

## 🔬 Test-Driven Development (TDD) Methodology

This project was built using **strict TDD workflow** across 5 development cycles:

### TDD Cycle Overview

| Cycle | Module | Tests | Focus |
|-------|--------|-------|-------|
| **Cycle 1** | Data Validation | 8 | Schema validation, null checks, required columns |
| **Cycle 2** | Preprocessing | 10 | Feature scaling, one-hot encoding, output validation |
| **Cycle 3** | Feature Engineering | 12 | Tenure groups, derived features, edge cases |
| **Cycle 4** | Model Training | 12 | Model fitting, reproducibility, input handling |
| **Cycle 5** | Prediction Interface | 14 | Single/batch predictions, probabilities, error handling |
| **Bonus** | Leakage-Safe Preprocessing | 8 | Fit/transform separation, no refitting on test data |

### TDD Workflow

Each cycle followed the **RED → GREEN → REFACTOR** pattern:

1. **RED Phase** 🔴
   - Write failing tests first
   - Tests define expected behavior
   - Screenshot of failing tests saved to `outputs/`

2. **GREEN Phase** 🟢
   - Implement minimal code to pass tests
   - All tests must pass
   - Screenshot of passing tests saved to `outputs/`

3. **REFACTOR Phase** 🔵
   - Clean up code
   - Improve readability
   - Tests remain passing

### Why TDD?

- **Prevents regressions** - Changes that break functionality are caught immediately
- **Living documentation** - Tests describe how code should behave
- **Design quality** - Writing tests first leads to better API design
- **Confidence** - 64 passing tests provide confidence in correctness

---

## 🤖 Model & Evaluation

### Model Choice: RandomForest Classifier

**Why RandomForest?**
- ✅ **Interpretable** - Feature importance readily available
- ✅ **Robust** - Handles non-linear relationships and feature interactions
- ✅ **No feature scaling required** - Works with mixed feature types
- ✅ **Baseline performance** - Strong out-of-the-box results
- ✅ **Ensemble method** - Reduces overfitting through averaging

**Hyperparameters:**
```python
RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    max_depth=10,          # Limit tree depth to prevent overfitting
    min_samples_split=5,   # Minimum samples to split a node
    min_samples_leaf=2,    # Minimum samples in leaf node
    random_state=42,       # Reproducibility
    n_jobs=-1              # Use all CPU cores
)
```

### Evaluation Metrics

The pipeline evaluates model performance using multiple metrics:

- **Accuracy** - Overall correctness
- **Precision** - Of predicted churners, how many actually churned?
- **Recall** - Of actual churners, how many did we catch?
- **F1 Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve (threshold-independent)
- **Confusion Matrix** - Breakdown of predictions vs. actuals
- **Classification Report** - Per-class precision, recall, F1

**Metrics are calculated on held-out test set (20% of data)** to provide unbiased performance estimates.

### Data Leakage Prevention

**Critical:** This project implements **leakage-safe preprocessing** to ensure test data statistics do not influence training:

```python
# ✅ CORRECT: Fit on train, transform on test
preprocessor, X_train = fit_preprocess(train_df)
X_test = transform_preprocess(test_df, preprocessor)

# ❌ WRONG: Fitting on test data leaks information
X_test = preprocess_data(test_df)  # Refits on test data!
```

**8 dedicated tests** validate that preprocessing does not refit on test data.

---

## 🔍 Explainability & Interpretability

### Feature Engineering for Interpretability

The project creates **domain-meaningful features** that are easy to explain:

1. **Tenure Groups** - Categorical buckets (new, short, medium, long, very_long)
   - Business insight: "Long-tenure customers are less likely to churn"

2. **Average Monthly Charge** - `TotalCharges / tenure`
   - Business insight: "Customers paying more per month may have higher expectations"

3. **Charge Ratio** - `MonthlyCharges / TotalCharges`
   - Business insight: "Recent price increases may trigger churn"

### RandomForest Feature Importance

RandomForest provides built-in feature importance scores:

```python
# Extract feature importance
importances = model.feature_importances_
feature_names = [...] # Feature names from preprocessing

# Sort and display top features
top_features = sorted(zip(feature_names, importances), 
                     key=lambda x: x[1], reverse=True)[:10]
```

**Interpretation:** Features with higher importance have more influence on churn predictions.

### Limitations of Current Explainability

- ❌ **No SHAP/LIME** - Individual prediction explanations not implemented
- ❌ **No feature importance visualization** - Results not plotted
- ❌ **No partial dependence plots** - Feature effect curves not generated

**Next steps:** Add SHAP values for instance-level explanations.

---

## ⚠️ Limitations & Known Issues

### Current Limitations

1. **Preprocessing Leakage in Legacy Function**
   - `preprocess_data()` uses `fit_transform` internally
   - **Mitigation:** New `fit_preprocess()` and `transform_preprocess()` functions added
   - **Status:** Pipeline uses leakage-safe functions; legacy function retained for backward compatibility

2. **No Model Persistence**
   - Trained models are not saved to disk
   - **Impact:** Model must be retrained for each use
   - **Next step:** Add `joblib` or `pickle` for model serialization

3. **No Hyperparameter Tuning**
   - RandomForest uses default hyperparameters
   - **Impact:** Potential performance left on table
   - **Next step:** Implement GridSearchCV or RandomizedSearchCV

4. **Limited Explainability**
   - Feature importance available but not extracted/visualized
   - No SHAP or LIME for instance-level explanations
   - **Next step:** Add explainability module with visualizations

5. **No Production Deployment**
   - No API endpoint or serving infrastructure
   - **Next step:** Add Flask/FastAPI REST API for predictions

6. **Single Model**
   - Only RandomForest implemented
   - **Next step:** Compare with XGBoost, LightGBM, Logistic Regression

### Data Assumptions

- Dataset is assumed to be clean (no major outliers or data quality issues)
- Target variable ('Churn') is balanced or stratified sampling is sufficient
- Features are assumed to be relevant and not redundant

---

## 🎯 Next Steps & Future Improvements

### High Priority

1. **Add Model Persistence**
   ```python
   import joblib
   joblib.dump(model, 'models/churn_model.pkl')
   ```

2. **Extract & Visualize Feature Importance**
   ```python
   import matplotlib.pyplot as plt
   # Plot top 10 features
   ```

3. **Add SHAP Explanations**
   ```python
   import shap
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X_test)
   ```

4. **Hyperparameter Tuning**
   ```python
   from sklearn.model_selection import GridSearchCV
   # Tune n_estimators, max_depth, min_samples_split
   ```

### Medium Priority

5. **Add CI/CD Pipeline**
   - GitHub Actions to run tests on every push
   - Automated test coverage reporting

6. **Model Comparison**
   - Implement XGBoost, LightGBM, Logistic Regression
   - Compare performance metrics

7. **Cross-Validation**
   - Add k-fold cross-validation for robust performance estimates

8. **Data Profiling**
   - Use `pandas-profiling` to generate EDA report

### Low Priority

9. **REST API**
   - Flask or FastAPI endpoint for predictions
   - Docker containerization

10. **Monitoring & Logging**
    - Add structured logging
    - Track prediction latency and model drift

---

## 📊 Test Coverage

**Total Tests:** 64  
**Test Coverage:** Comprehensive

| Module | Tests | Coverage |
|--------|-------|----------|
| Data Validation | 8 | Schema, nulls, required columns |
| Preprocessing | 18 | Encoding, scaling, leakage prevention |
| Feature Engineering | 12 | Derived features, edge cases |
| Model Training | 12 | Fitting, reproducibility, input types |
| Prediction | 14 | Single/batch, probabilities, errors |

**All tests use synthetic data** for isolation and fast execution.

---

## 🤝 Contributing

This project was built as a portfolio/academic project demonstrating:
- Production-quality ML engineering
- Strict TDD methodology
- Data leakage prevention
- Clean code architecture

For questions or suggestions, please open an issue.

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 👤 Author

**Sourabha KK**
- GitHub: [@SourabhaKK](https://github.com/SourabhaKK)
- Project: [customer-churn-prediction-ml](https://github.com/SourabhaKK/customer-churn-prediction-ml)

---

## 🙏 Acknowledgments

- **TDD Methodology** - Inspired by Kent Beck's "Test-Driven Development by Example"
- **Scikit-learn** - Excellent ML library with clear documentation
- **Pytest** - Powerful and intuitive testing framework

---

**Built with ❤️ using Test-Driven Development**
