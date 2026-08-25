import joblib

# 1. Define the exact feature names used in training & inference
feature_columns = [
    'Customer Service Calls',
    'Lifetime Value ($)',
    'Cart Abandonment Rate (%)',
    'Login Frequency (per month)',
    'Avg Session Duration (min)',
    'Membership Years'
]

# 2. Save to feature_columns.pkl
joblib.dump(feature_columns, 'feature_columns.pkl')

print("✅ 'feature_columns.pkl' created successfully!")