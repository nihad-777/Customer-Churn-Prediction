import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
import joblib

# 1. Load dataset & artifacts
df = pd.read_csv('ecommerce_churn.csv')
feature_columns = joblib.load('feature_columns.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Prepare X and y
X = df[feature_columns]
y = df['Churn']  # Ensure target column is named 'Churn'

# 3. Split & scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")

# 6. Save to churn_model.pkl
joblib.dump(model, 'churn_model.pkl')

print("✅ 'churn_model.pkl' created successfully!")