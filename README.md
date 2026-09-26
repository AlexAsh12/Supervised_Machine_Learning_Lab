# Supervised Machine Learning – Wine Quality & Breast Cancer Classification

This repository contains a Jupyter notebook completed as part of a Supervised Machine Learning lab at ESILV (École Supérieure d'Ingénieurs Léonard de Vinci), covering regression, classification, and model evaluation on two real-world datasets.

## Contents

The notebook (`LAB2_Alexandre_Folleas_Supervised_ML.ipynb`) is organized into 6 exercises:

1. **Data exploration** – Loading and exploring the Wine Quality dataset (physicochemical properties of red wine).
2. **Regression pipelines** – Building `scikit-learn` pipelines with `StandardScaler` / `MinMaxScaler` and comparing their effect on model predictions (R², MSE).
3. **Binary classification on imbalanced data** – Recasting the wine quality problem as a binary classification task ("good" vs "bad" wine, 86.6% / 13.4% split) using logistic regression.
4. **Beyond accuracy** – Analyzing why accuracy is misleading on imbalanced data, and using precision, recall, and F1-score instead.
5. **Model comparison via ROC-AUC** – Comparing logistic regression and Naive Bayes with stratified 5-fold cross-validation.
6. **Breast cancer classification** – Applying the same methodology to the Wisconsin Breast Cancer dataset (malignant vs. benign tumor classification), with a discussion of recall as the key metric in a medical diagnosis context.

## Tools and libraries

- Python, Jupyter Notebook
- `pandas`, `numpy` for data manipulation
- `scikit-learn` (`LogisticRegression`, `Pipeline`, `StandardScaler`, `StratifiedKFold`, `cross_validate`, etc.) for modeling and evaluation
- `matplotlib` for visualization (ROC curves, distributions)

## Key results

- On the breast cancer dataset, logistic regression outperformed Naive Bayes (ROC-AUC 0.993 vs 0.988; accuracy 0.974 vs 0.933), with only 1 false negative on the test set vs 4 for Naive Bayes.
- On the imbalanced wine quality dataset, the analysis shows why accuracy alone is not a reliable metric and motivates the use of precision/recall/F1 and ROC-AUC.

## Author

Alexandre Folleas – ESILV, MSc in Financial Engineering
