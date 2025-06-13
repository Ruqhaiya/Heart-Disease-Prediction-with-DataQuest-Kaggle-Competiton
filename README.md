# Heart-Disease-Prediction-with-DataQuest-Kaggle-Competiton

# Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Methodology](#methodology)
- [Computational Results](#computational-results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [References](#references)

## Abstract
In this project, we develop and evaluate machine‐learning pipelines to predict the presence of heart disease using a curated dataset of 918 patients drawn from five legacy cohorts. We combine robust preprocessing—handling physiologically implausible zeros in cholesterol and blood pressure via median and group‐wise imputation—with targeted feature engineering (age buckets, ST‐depression under exertion, cholesterol/blood‐pressure ratio). We benchmark Logistic Regression, Support Vector Machines, Random Forests, and SMOTE‐augmented KNN inside nested cross‐validation loops. Our tuned Random Forest achieves 0.932 ROC‐AUC and 0.880 accuracy on the public leaderboard, placing 9th out of 39 teams.

## Introduction
Cardiovascular diseases remain the leading cause of mortality worldwide. Early identification of at‐risk individuals allows for timely intervention and prevention of adverse outcomes. Kaggle’s Heart Disease Prediction challenge provides a real‐world dataset of demographic, clinical, and exercise test results to develop binary classifiers for heart disease presence. This notebook walks through data cleaning, exploratory analysis, feature construction, model building, and rigorous evaluation, all encapsulated in reproducible scikit‐learn pipelines.

## Theoretical Background
The task of binary classification in medical diagnostics entails assigning observations to one of two discrete categories—here, presence or absence of cardiovascular disease—based on a vector of patient‐level features. Supervised learning algorithms achieve this by estimating a decision boundary or probabilistic mapping $f: \mathbb{R}^d \to \{0,1\}$ that minimizes misclassification risk on unseen data. However, medical datasets often exhibit characteristics such as class imbalance, heterogeneous feature scales, and non‐linear interactions, which necessitate specialized methodological considerations.

### Logistic Regression and Regularization

Logistic regression models the log‐odds of the positive class as a linear combination of input features:

$$
\log\frac{P(y=1\mid \mathbf{x})}{P(y=0\mid \mathbf{x})} = \beta_0 + \sum_{j=1}^d \beta_j x_j.
$$

Maximum‐likelihood estimation of $\boldsymbol{\beta}$ is convex but can overfit when $d$ is large or features are collinear. Regularization penalties such as the $\ell_1$ (Lasso) and $\ell_2$ (Ridge) norms augment the loss with $\alpha \|\boldsymbol{\beta}\|_1$ or $ \frac{\alpha}{2}\|\boldsymbol{\beta}\|_2^2$, improving generalization and enabling implicit feature selection \[@tibshirani1996regression; @hoerl1970ridge].

### Decision Trees and Ensemble Methods

Decision trees partition the feature space via axis-aligned splits that maximize information gain or Gini impurity reduction. While highly interpretable, single trees are prone to high variance and overfitting. Random forests mitigate this by averaging predictions over an ensemble of decorrelated trees, each trained on a bootstrap sample and a random subset of features \[@breiman2001random]. This Bagging strategy reduces variance without substantially increasing bias, making RF robust to noisy measurements and complex feature interactions.

### K–Nearest Neighbors (KNN) and Distance Metrics

KNN is a non‐parametric, instance‐based learner that classifies a query point by majority vote among its $k$ nearest neighbors under a chosen metric (e.g., Euclidean or Manhattan distance). Its performance is sensitive to feature scaling, the choice of $k$, and the intrinsic dimensionality of the data. High‐dimensional or sparsely sampled spaces can degrade KNN’s effectiveness due to the “curse of dimensionality” \[@bellman1961adaptive].

### Handling Class Imbalance: SMOTE

Medical datasets often exhibit skewed class distributions, where positive cases (disease) are underrepresented. Synthetic Minority Over‐sampling Technique (SMOTE) addresses this by generating new minority‐class samples via interpolation between nearest neighbors in feature space \[@chawla2002smote]. When integrated within cross‐validation folds (“in‐fold SMOTE”), this prevents leakage of synthetic examples into evaluation sets and yields more reliable performance estimates.

### Model Selection and Nested Cross‐Validation

Hyperparameter tuning and model selection introduce optimism bias if the same data are used for both parameter optimization and evaluation. Nested cross‐validation structures an outer loop to estimate generalization error and an inner loop to select hyperparameters, thereby providing an nearly unbiased estimate of performance \[@varma2006bias]. This approach is critical when selecting among complex pipelines combining preprocessing, sampling strategies, and varied learner classes.

### Feature Engineering and Collinearity

Domain‐informed feature transformations—such as interaction terms (e.g., ST‐depression × angina) or composite ratios (cholesterol / blood pressure)—can capture non‐linear relationships that raw measurements do not. However, high collinearity among raw and engineered variables can inflate variance in linear models and obscure feature importance. Pruning redundant features based on correlation thresholds or regularized feature‐selection methods (e.g., $\ell_1$ regularization) helps maintain model parsimony and interpretability \[@friedman2001elements].

## Methodology
1. **Data Cleaning**: Convert zeros in `Cholesterol` and `RestingBP` to `NaN`, impute train‐set missing values using class‐specific medians, and test‐set missing values using overall medians.  
2. **Feature Engineering**: Create `Age_bin` (decade buckets), `Old_angina` (ST‐depression × exercise‐induced angina), and `Chol_BP_ratio` (cholesterol/resting BP).  
3. **EDA**: Visualize distributions, boxplots, and countplots to confirm that older age, lower maximum heart rate, higher ST‐depression, asymptomatic chest pain, and flat/down‐sloping ST contours correlate strongly with disease.  
4. **Pipeline Construction**: Use `ColumnTransformer` for median imputation, standard scaling of numerics, and one‐hot encoding of categoricals (with `drop='first'`).  
5. **Modeling & Evaluation**: Compare Logistic Regression, SVM, Random Forest, and SMOTE‐KNN in nested 5-fold cross‐validation, tuning hyperparameters via `RandomizedSearchCV` for RF and `GridSearchCV` for KNN and SVM.

## Computational Results
- **Baseline Logistic Regression**: 0.846 ± 0.029 accuracy, 0.922 ± 0.035 AUC  
- **Support Vector Machine**: 0.857 ± 0.017 accuracy  
- **Random Forest (default)**: 0.861 ± 0.026 accuracy, 0.932 ± 0.025 AUC  
- **Tuned Random Forest** (RandomizedSearch): Achieves 0.87+ accuracy on nested CV  
- **SMOTE-augmented KNN**: ~0.864 ± 0.024 accuracy  

On the public leaderboard, our final Random Forest submission scored **0.880 accuracy** and **0.932 AUC**, ranking **9th out of 39 teams**.

## Discussion
Our results demonstrate that tree‐based ensembles outperform linear and kernel methods on this moderate-sized heart‐disease dataset. Feature engineering—particularly ST‐depression under exercise and the cholesterol/BP ratio—was critical. SMOTE improved recall without sacrificing precision in KNN, though RF remained more stable. Perfect test‐set labels were hidden; we relied on nested CV and a held‐out split for honest performance estimation. Future work could explore metric learning (e.g. Neighborhood Components Analysis), deep feature interactions (e.g. gradient boosting with monotonic constraints), or domain adaptation for external cohorts.

## Conclusion
We’ve built a comprehensive, end-to-end ML solution that balances interpretability and predictive power, using best practices in data cleaning, nested cross‐validation, and hyperparameter optimization. Our tuned Random Forest pipeline delivers strong performance on the heart‐disease prediction task and serves as a template for similar medical classification problems.

## References

1. **Pastor Soto.** *Heart Disease Prediction with Dataquest*. Kaggle Competition: [https://www.kaggle.com/competitions/heart-disease-prediction-dataquest/overview](https://www.kaggle.com/competitions/heart-disease-prediction-dataquest/overview) ([kaggle.com][1])
2. **World Health Organization.** *Cardiovascular diseases (CVDs)*. Fact sheet: [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29) ([who.int][2])
3. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.
4. **Pedregosa, F., Varoquaux, G., Gramfort, A., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
5. **Friedman, J., Hastie, T., & Tibshirani, R.** (2009). *The Elements of Statistical Learning*. Springer.
