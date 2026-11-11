# ============================================================
# BLINKIT SALES ANALYSIS - COMPLETE MACHINE LEARNING PROJECT
# Data Cleaning | EDA | Feature Engineering | Model Optimization
# Linear Regression | Random Forest | XGBoost | Voting Ensemble
# ============================================================

# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split,
    KFold
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.inspection import permutation_importance


# 2. LOAD DATASET
# ============================================================

df = pd.read_excel("D:/Project2/Blinkit_dataset.xlsx")

print("\n========== DATASET OVERVIEW ==========")
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())


# 3. DATA CLEANING & DOMAIN-SPECIFIC IMPUTATION
# ============================================================

# Standardize fat content
df['Item_Fat_Content'] = (
    df['Item_Fat_Content']
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({
        'lf': 'Low Fat',
        'low fat': 'Low Fat',
        'reg': 'Regular'
    })
)

# Extract 2-letter Category Code (FD = Food, DR = Drinks, NC = Non-Consumable)
df['Item_Category_Code'] = df['Item_Identifier'].astype(str).str[:2]
# Non-consumable items should have Non-Edible fat content
df.loc[df['Item_Category_Code'] == 'NC', 'Item_Fat_Content'] = 'Non-Edible'

# Impute Item_Weight using the mean of the exact Item_Identifier
item_weight_mean = df.groupby('Item_Identifier')['Item_Weight'].transform('mean')
df['Item_Weight'] = df['Item_Weight'].fillna(item_weight_mean).fillna(df['Item_Weight'].median())

# Handle 0 visibility by calculating the item's mean visibility across outlets
item_vis_mean = df.groupby('Item_Identifier')['Item_Visibility'].transform(
    lambda s: s.replace(0, np.nan).mean()
)
df['Item_Visibility'] = (
    df['Item_Visibility']
    .replace(0, np.nan)
    .fillna(item_vis_mean)
    .fillna(df['Item_Visibility'].median())
)

# Impute Outlet_Size using mode within each Outlet_Type
mode_size_by_type = df.groupby('Outlet_Type')['Outlet_Size'].apply(
    lambda s: s.mode()[0] if not s.mode().empty else 'Small'
)
df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Type'].map(mode_size_by_type))

# Remove records missing the target
df = df.dropna(subset=['Item_Outlet_Sales'])

print("\nDuplicate Records:", df.duplicated().sum())
print("Cleaned Data Shape:", df.shape)


# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

# Outlet Type vs Sales
plt.figure(figsize=(10, 5))
sns.barplot(
    x='Outlet_Type',
    y='Item_Outlet_Sales',
    data=df,
    errorbar=None
)
plt.xticks(rotation=45)
plt.title("Average Sales by Outlet Type")
plt.tight_layout()
plt.show()

# Item MRP vs Sales
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='Item_MRP',
    y='Item_Outlet_Sales',
    data=df,
    alpha=0.3
)
plt.title("Item MRP vs Sales")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 7))
corr = df.corr(numeric_only=True)
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Sales Distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    df['Item_Outlet_Sales'],
    bins=40,
    kde=True
)
plt.title("Sales Distribution")
plt.tight_layout()
plt.show()


# 5. FEATURE ENGINEERING
# ============================================================

# Historical reference year for outlet age
REFERENCE_YEAR = 2022
df['Outlet_Age'] = REFERENCE_YEAR - df['Outlet_Establishment_Year']

# MRP Price Bands
df['Item_MRP_Band'] = pd.cut(
    df['Item_MRP'],
    bins=[-np.inf, 70, 140, 200, np.inf],
    labels=[
        'Budget',
        'Economy',
        'Premium',
        'High Premium'
    ]
)

# Visibility Ratio (item visibility compared to that item's average across stores)
df['Item_Visibility_Ratio'] = df['Item_Visibility'] / (item_vis_mean + 1e-6)

print("\nFeature Engineering Completed")


# 6. FEATURE AND TARGET SEPARATION
# ============================================================

# Note: We drop Item_Identifier to avoid 1500+ high-cardinality one-hot columns
# and drop Outlet_Establishment_Year since Outlet_Age captures the variance.
features_to_drop = ['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Establishment_Year']

X = df.drop(columns=features_to_drop)
y = df['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", X_train.shape)
print("Testing Records:", X_test.shape)


# 7. PREPROCESSING PIPELINE
# ============================================================

numeric_features = X.select_dtypes(
    include=['number']
).columns.tolist()

categorical_features = X.select_dtypes(
    include=['object', 'category', 'bool', 'string']
).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# 8. MODEL EVALUATION FUNCTION
# ============================================================

results = []

def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print("\n================================")
    print(name)
    print("================================")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })
    return r2


# 9. LINEAR REGRESSION & RIDGE REGRESSION
# ============================================================

lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
evaluate_model("Linear Regression", y_test, lr_pred)

ridge_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=10.0))
])
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
evaluate_model("Ridge Regression", y_test, ridge_pred)


# 10. RANDOM FOREST (OPTIMIZED DEPTH)
# ============================================================

rf_reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_reg)
])
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
evaluate_model("Random Forest", y_test, rf_pred)


# 11. GRADIENT BOOSTING REGRESSOR
# ============================================================

gbr_reg = GradientBoostingRegressor(
    n_estimators=80,
    learning_rate=0.08,
    max_depth=3,
    random_state=42
)

gbr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', gbr_reg)
])
gbr_model.fit(X_train, y_train)
gbr_pred = gbr_model.predict(X_test)
evaluate_model("Gradient Boosting", y_test, gbr_pred)


# 12. TUNED XGBOOST REGRESSOR
# ============================================================

xgb_reg = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=75,
    max_depth=3,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_reg)
])
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
evaluate_model("Tuned XGBoost", y_test, xgb_pred)


# 13. VOTING ENSEMBLE REGRESSOR (BEST OVERALL)
# ============================================================

ensemble_reg = VotingRegressor(
    estimators=[
        ('rf', rf_reg),
        ('gbr', gbr_reg),
        ('xgb', xgb_reg)
    ],
    weights=[1.2, 1.0, 1.0]
)

ensemble_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', ensemble_reg)
])
ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)
evaluate_model("Voting Ensemble (RF + GBR + XGB)", y_test, ensemble_pred)


# 14. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='R2', ascending=False)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# R2 Comparison Plot
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='Model', y='R2')
plt.xticks(rotation=30)
plt.title("R2 Comparison Across Models")
plt.tight_layout()
plt.show()

# MAE Comparison Plot
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='Model', y='MAE')
plt.xticks(rotation=30)
plt.title("MAE Comparison Across Models")
plt.tight_layout()
plt.show()


# 15. XGBOOST FEATURE IMPORTANCE
# ============================================================

feature_names = (
    xgb_model.named_steps['preprocessor']
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_reg.feature_importances_
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n========== TOP 15 XGBOOST FEATURES ==========")
print(importance_df.head(15).to_string(index=False))

# Feature Importance Plot
top_features = importance_df.head(15)
plt.figure(figsize=(11, 7))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.gca().invert_yaxis()
plt.title("Top 15 XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# 16. PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating Permutation Importance on Best Model...")

perm = permutation_importance(
    ensemble_model,
    X_test,
    y_test,
    scoring='r2',
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': perm.importances_mean,
    'Std': perm.importances_std
})

perm_df = perm_df.sort_values(by='Importance', ascending=False)

print("\n========== PERMUTATION IMPORTANCE ==========")
print(perm_df.to_string(index=False))

# Permutation Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(perm_df['Feature'], perm_df['Importance'], xerr=perm_df['Std'])
plt.gca().invert_yaxis()
plt.title("Permutation Feature Importance (Voting Ensemble)")
plt.xlabel("Mean decrease in R2")
plt.tight_layout()
plt.show()


# 17. ACTUAL VS PREDICTED SALES & RESIDUAL ANALYSIS
# ============================================================

best_pred = ensemble_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_pred, alpha=0.3)
min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales - Best Model (Voting Ensemble)")
plt.tight_layout()
plt.show()

residuals = y_test.values - best_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(best_pred, residuals, alpha=0.3)
plt.axhline(y=0, linestyle='--', color='red')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Sales")
plt.tight_layout()
plt.show()


# 18. SAVE RESULTS
# ============================================================

output_path = "D:/Project2/"

results_df.to_csv(output_path + "model_comparison.csv", index=False)
importance_df.to_csv(output_path + "xgboost_feature_importance.csv", index=False)
perm_df.to_csv(output_path + "permutation_importance.csv", index=False)

prediction_df = pd.DataFrame({
    'Actual_Sales': y_test.values,
    'Predicted_Sales': best_pred,
    'Residual': residuals
})
prediction_df.to_csv(output_path + "best_model_predictions.csv", index=False)


# 19. FINAL SUMMARY
# ============================================================

print("\n\n==========================================")
print("BLINKIT SALES ANALYSIS - FINAL SUMMARY")
print("==========================================")
print("\nBest Model:")
print(results_df.iloc[0]['Model'])

print("\nBest Test R2 Score:")
print(round(results_df.iloc[0]['R2'], 4))

print("\nBest Model MAE:")
print(round(results_df.iloc[0]['MAE'], 2))

print("\nBest Model RMSE:")
print(round(results_df.iloc[0]['RMSE'], 2))

print("\nTop Predictive Feature:")
print(importance_df.iloc[0]['Feature'])

print("\nAll Results and Charts Exported Successfully!")

# ============================================================
# END OF PROJECT
# ============================================================
