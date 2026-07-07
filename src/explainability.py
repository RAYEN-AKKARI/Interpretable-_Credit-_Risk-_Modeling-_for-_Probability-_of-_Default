import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(feature_names, importance_values, title="Feature Importance"):
    """
    Plot feature importance values.
    """
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

    return importance_df


def get_logistic_coefficients(model, feature_names):
    """
    Return logistic regression coefficients as a sorted DataFrame.
    """
    coefficients = model.coef_[0]

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()

    coef_df = coef_df.sort_values(
        by="Abs_Coefficient",
        ascending=False
    )

    return coef_df
