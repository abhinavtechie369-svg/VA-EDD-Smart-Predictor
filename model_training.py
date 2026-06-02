import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
model = MultiOutputRegressor(RandomForestRegressor())

# Load data
df = pd.read_csv("data/va_edd_dataset.csv")

# Feature Engineering
df['IP_TON'] = df['IP'] * df['TON']
df['TON_TOFF'] = df['TON'] * df['TOFF']
df['IP_VA'] = df['IP'] * df['VA']

# Inputs & Outputs
X = df[['IP','TON','TOFF','TR','VA','IP_TON','TON_TOFF','IP_VA']]
y = df[['MRR','SR']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = MultiOutputRegressor(Ridge(alpha=1.0))

# Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []
mse_scores = []

for train_idx, test_idx in kf.split(X_scaled):
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2_scores.append(r2_score(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))

print("R2 Score:", np.mean(r2_scores))
print("MSE:", np.mean(mse_scores))

# 🔥 Coefficients (MODEL BRAIN)
model.fit(X_scaled, y)

ridge_mrr = model.estimators_[0]
ridge_sr = model.estimators_[1]

print("\nMRR Coefficients:", ridge_mrr.coef_)
print("SR Coefficients:", ridge_sr.coef_)