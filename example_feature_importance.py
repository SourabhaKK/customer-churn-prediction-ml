"""
Example: Feature Importance Extraction

This script demonstrates how to extract and display feature importance
from the trained RandomForest model.
"""

from src.pipeline import run_pipeline
from src.evaluate import get_feature_importance

# Run the full pipeline
print("Running pipeline...")
model, metrics = run_pipeline('data/customer_churn_dataset.csv')

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1 Score:  {metrics.get('f1', 'N/A'):.4f}")
print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

# Extract feature importance
# Note: Feature names would need to be tracked through the pipeline
# For now, we'll use generic names
print("\n" + "="*60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*60)

try:
    # Get top 10 features
    top_features = get_feature_importance(model, feature_names=None, top_n=10)
    
    for i, (feature, importance) in enumerate(top_features, 1):
        bar = "█" * int(importance * 50)  # Visual bar
        print(f"{i:2d}. {feature:20s} {importance:.4f} {bar}")
        
except AttributeError as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("Features with higher importance scores have more influence on")
print("churn predictions. Focus retention efforts on customers with")
print("unfavorable values in these top features.")
print("="*60)
