# Model Card

## Model Card for "The Impact of Dune Plants on Caretta Caretta Nest Success" Model

### Overview
This project investigates how dune plant roots affect the nesting success of loggerhead sea turtles (Caretta caretta). Since these turtles lay eggs near dunes, roots can damage the eggs, leading to reduced hatching success. The goal is to analyze and optimize the impact of dune plants on hatching success using predictive modeling.

### Project Details
- **Project Name:** The Impact of Dune Plants on Caretta Caretta Nest Success
- **Team Members:**  Ceren Kılıç, Mihriban Özdemir, Türkan Rişvan
- **Institution:** Samsung Innovation Campus
- **Related SDG:** SDG-14 (Life Below Water)

### Problem Statement
The nesting success of loggerhead sea turtles is influenced by the presence of dune plant roots, which can break the eggs and reduce hatching success. Analyzing and optimizing these effects are crucial for improving hatching rates and supporting conservation efforts.

### Dataset
- **Source:** Dryad data repository
- **Collection Year:** 2022
- **Features:** 42 (combination of categorical and numerical data)
- **Samples:** 93 nests
- **Monitoring Duration:** 6 months
- **Reference Paper:** Redding, O. T., Castorani, M. C., & Lasala, J. (2024). The effects of dune plant roots on loggerhead turtle (Caretta caretta) nest success. Ecology and Evolution, 14(4), e11207. https://doi.org/10.1002/ece3.11207

### Model Details
#### Model Description
The model predicts the number of live and dead eggs in loggerhead turtle nests based on various environmental and biological factors. Regression models, including logistic regression, decision trees, random forests, and XGBoost, are used to predict nest success.

#### Key Steps
1. **Data Collection and Preprocessing:**
   - Handling missing values via imputation or exclusion.
   - Normalizing and standardizing data for comparability.
2. **Exploratory Data Analysis (EDA):**
   - Using histograms and scatter plots to identify patterns.
3. **Model Building:**
   - Implementing regression models to predict nest success.
   - Evaluating models using cross-validation and metrics such as accuracy, precision, recall, and F1-score.
4. **Model Validation and Improvement:**
   - Hyperparameter tuning and feature selection using Recursive Feature Elimination (RFE).
   - Comparing model performance before and after optimization.

### Model Performance
#### Initial Model Scores
- **Logistic Regression (LR):**
  - Train: MAE: 0.089, MSE: 0.025, RMSE: 0.159, R²: 0.999
  - Test: MAE: 0.396, MSE: 0.401, RMSE: 0.634, R²: 0.999
- **Support Vector Regression (SVR):**
  - Train: MAE: 16.238, MSE: 751.870, RMSE: 27.420, R²: -0.066
  - Test: MAE: 15.150, MSE: 660.680, RMSE: 25.704, R²: -0.087
- **Decision Tree (CART):**
  - Train: MAE: 3.773e-16, MSE: 5.361e-30, RMSE: 2.315e-15, R²: 1.0
  - Test: MAE: 0.886, MSE: 3.576, RMSE: 1.891, R²: 0.994
- **Random Forest (RF):**
  - Train: MAE: 0.639, MSE: 1.401, RMSE: 1.184, R²: 0.998
  - Test: MAE: 1.228, MSE: 4.978, RMSE: 2.231, R²: 0.992
- **Elastic Net (ENet):**
  - Train: MAE: 5.081, MSE: 68.032, RMSE: 8.248, R²: 0.904
  - Test: MAE: 4.688, MSE: 56.955, RMSE: 7.547, R²: 0.906
- **XGBoost Regressor (XGBRegressor):**
  - Train: MAE: 0.00039, MSE: 3.594e-07, RMSE: 0.0006, R²: 1.0
  - Test: MAE: 0.542, MSE: 2.785, RMSE: 1.669, R²: 0.995

#### Post Hyperparameter Optimization Scores
- **Logistic Regression (LR):**
  - Neg. MSE: -952.112, Neg. MAE: -9.270, R²: -0.902
  - RFE (20 features): Neg. MSE: -206.827, Neg. MAE: -1.821, R²: 0.571
- **Support Vector Regression (SVR):**
  - Before: Neg. MSE: -732.705, Neg. MAE: -16.609, R²: -0.100
  - After: Neg. MSE: -3.164, Neg. MAE: -1.180, R²: 0.994
  - RFE (20 features): Neg. MSE: -2.068, Neg. MAE: -0.791, R²: 0.996
- **Decision Tree (CART):**
  - Before: Neg. MSE: -39.730, Neg. MAE: -4.469, R²: 0.889
  - After: Neg. MSE: -28.595, Neg. MAE: -3.654, R²: 0.931
  - RFE (20 features): Neg. MSE: -28.845, Neg. MAE: -3.686, R²: 0.931
- **Random Forest (RF):**
  - Before: Neg. MSE: -23.311, Neg. MAE: -3.159, R²: 0.952
  - After: Neg. MSE: -20.945, Neg. MAE: -3.010, R²: 0.957
  - RFE (20 features): Neg. MSE: -19.649, Neg. MAE: -2.941, R²: 0.958
- **Elastic Net (ENet):**
  - Before: Neg. MSE: -105.097, Neg. MAE: -6.336, R²: 0.855
  - After: Neg. MSE: -1.049, Neg. MAE: -0.619, R²: 0.998
- **XGBoost Regressor (XGBRegressor):**
  - Before: Neg. MSE: -60.000, Neg. MAE: -4.088, R²: 0.893
  - After: Neg. MSE: -31.979, Neg. MAE: -3.456, R²: 0.956
  - RFE (20 features): Neg. MSE: -50.312, Neg. MAE: -4.019, R²: 0.919

### References
1. **Dataset:** Redding, Olivia; Castorani, Max; Lasala, Jake (2024). Case study data 2022: The effects of dune plant roots on loggerhead turtle (Caretta caretta) nest success [Dataset]. Dryad. https://doi.org/10.5061/dryad.zw3r228dk
2. **Paper:** Redding, O. T., Castorani, M. C., & Lasala, J. (2024). The effects of dune plant roots on loggerhead turtle (Caretta caretta) nest success. Ecology and Evolution, 14(4), e11207. https://doi.org/10.1002/ece3.11207
