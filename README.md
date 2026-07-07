# Interpretable Credit Risk Modeling for Probability of Default

This project focuses on building interpretable machine learning models for **credit risk assessment** and **Probability of Default (PD) prediction**.

The objective is to compare predictive performance and interpretability in a banking context, using models such as Logistic Regression, GLMBoost-inspired models and tree-based Gradient Boosting.

## Project context

Credit risk modeling is a key task in banking risk management. Financial institutions need to estimate the probability that a borrower may default while ensuring that the model remains understandable, explainable and aligned with regulatory expectations.

This project was developed as part of my second-year engineering project in Financial Modeling at ENIT.

## Main objective

The main objective is to study the trade-off between:

- Predictive performance
- Model interpretability
- Regulatory transparency
- Business usability in credit risk decision-making

## Models implemented

The project includes:

- Logistic Regression
- Custom GLMBoost-inspired implementation
- Tree-based Gradient Boosting model
- Feature importance analysis
- SHAP explanations
- LIME explanations

## Methodology

The workflow follows these steps:

1. Data understanding
2. Data preprocessing
3. Exploratory Data Analysis
4. Logistic Regression modeling
5. GLMBoost-inspired modeling
6. Tree-based Gradient Boosting comparison
7. Model evaluation
8. Model interpretability using SHAP and LIME
9. Comparative analysis of performance and explainability

## Evaluation metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC

## Key results

The comparison showed that tree-based Gradient Boosting achieved the best predictive performance, while Logistic Regression remained the most interpretable model.

The custom GLMBoost-inspired model provided an intermediate compromise between interpretability and predictive performance.

| Model | Accuracy | ROC-AUC | Recall Class 1 | Precision Class 1 | F1-score Class 1 |
|---|---:|---:|---:|---:|---:|
| Tree-based GLMBoost | 0.858 | 0.911 | 0.71 | 0.63 | 0.67 |
| Custom GLMBoost-inspired model | 0.836 | 0.895 | 0.46 | 0.63 | 0.53 |
| Logistic Regression | 0.826 | 0.898 | 0.49 | 0.58 | 0.53 |

## Interpretability

To improve model transparency, the project uses:

- Feature Importance
- SHAP for global explanations
- LIME for local explanations

These tools help identify the most influential variables and explain individual predictions.

## Data confidentiality

The original dataset used in the academic project comes from a professional banking environment and cannot be publicly shared.

For confidentiality reasons, this repository does not include private banking data. Future versions may include:

- A synthetic dataset
- A public credit risk dataset
- Reproducible notebooks adapted to open data

## Repository structure

```text
credit-risk-pd-modeling/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_custom_glmboost.ipynb
│   ├── 04_tree_based_glmboost.ipynb
│   └── 05_model_interpretability.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── logistic_model.py
│   ├── glmboost_model.py
│   ├── evaluation.py
│   └── explainability.py
│
├── reports/
│   └── figures/
│
├── models/
├── requirements.txt
├── LICENSE
└── README.md
```
## Tech stack

**Programming:** Python  
**Data analysis:** Pandas, NumPy  
**Machine learning:** Scikit-learn  
**Model interpretability:** SHAP, LIME  
**Visualization:** Matplotlib, Seaborn  
**Environment:** Jupyter Notebook

## Author

**Rayen Akkari**  
Financial Modeling Engineering Student at ENIT  
Interested in Data Science, Actuarial Science, Credit Risk and Quantitative Finance.
