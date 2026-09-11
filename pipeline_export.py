import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Sample Training Data
X_train = pd.DataFrame(
    {
        "tenure": [1, 12, 24, 48, 60],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 89.85],
        "TotalCharges": [29.85, 1889.50, 1081.25, 1840.75, 5681.10],
    }
)
y_train = [1, 0, 0, 0, 1]

# 2. Pipeline Creation (Preprocessing + Model)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["tenure", "MonthlyCharges", "TotalCharges"])
    ]
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# 3. Train & Save Pipeline
model_pipeline.fit(X_train, y_train)
joblib.dump(model_pipeline, "churn_pipeline.pkl")
print("✅ Pipeline Saved Successfully as 'churn_pipeline.pkl'!")