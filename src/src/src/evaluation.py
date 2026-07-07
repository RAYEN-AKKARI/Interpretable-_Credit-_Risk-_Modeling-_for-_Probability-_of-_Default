import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_classification_model(model, X_test, y_test, model_name: str):
    """
    Evaluate a binary classification model and return the main metrics.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        roc_auc = None

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision_Class_1": precision_score(y_test, y_pred, zero_division=0),
        "Recall_Class_1": recall_score(y_test, y_pred, zero_division=0),
        "F1_Class_1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc,
    }

    print(f"Model: {model_name}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return pd.DataFrame([metrics])
