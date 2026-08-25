import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load dataset (update filename if different)
df = pd.read_csv('ecommerce_churn.csv')

# 2. Load feature column list
feature_columns = joblib.load('feature_columns.pkl')

# 3. Fit the StandardScaler on the features
scaler = StandardScaler()
scaler.fit(df[feature_columns])

# 4. Save to scaler.pkl
joblib.dump(scaler, 'scaler.pkl')

print("✅ 'scaler.pkl' created successfully!")