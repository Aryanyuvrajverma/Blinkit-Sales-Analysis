import pandas as pd
df = pd.read_excel("D:/Project2/Blinkit_dataset.xlsx")
# Missing values
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())

df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

# Fix visibility
df['Item_Visibility'] = df['Item_Visibility'].replace(0, df['Item_Visibility'].median())

# Standardize categories
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
    'LF':'Low Fat', 'low fat':'Low Fat', 'reg':'Regular'
})
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x='Outlet_Type', y='Item_Outlet_Sales', data=df)
plt.xticks(rotation=45)
plt.show()

##df.to_csv("D:/Project2/blinkit_clean.csv", index=False)


import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

sns.scatterplot(x='Item_MRP', y='Item_Outlet_Sales', data=df, alpha=0.3)
plt.show()

#
##
###
####
# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

from sklearn.model_selection import train_test_split

X = df.drop('Item_Outlet_Sales', axis=1)
y = df['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, r2_score



lr_pred = lr.predict(X_test)
print("Linear Regression")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("R2:", r2_score(y_test, lr_pred))
print("------------")


pred = rf.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))
print("------------")

xgb_pred = xgb.predict(X_test)
print("XGBoost")
print("MAE:", mean_absolute_error(y_test, xgb_pred))
print("R2:", r2_score(y_test, xgb_pred))
print("------------")

import matplotlib.pyplot as plt

importance = rf.feature_importances_
features = X.columns
feat_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

# Sort and take top 10
top_features = feat_df.sort_values(by='Importance', ascending=False).head(10)

# Plot
plt.barh(top_features['Feature'], top_features['Importance'])
plt.gca().invert_yaxis()
plt.show()