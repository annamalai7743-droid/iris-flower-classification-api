import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

# 1. Dummy Churn Dataset உருவாக்கம் (Sample Data)
np.random.seed(42)
data = pd.DataFrame({
    'tenure': np.random.randint(1, 72, 100),
    'MonthlyCharges': np.random.uniform(20, 120, 100),
    'TotalCharges': np.random.uniform(100, 8000, 100),
    'Churn': np.random.choice([0, 1], 100)
})

X = data[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------------------
# 2. Global Feature Importance Chart
# -------------------------------------------------------------
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
feature_importances.plot(kind='barh', color='skyblue')
plt.title('Global Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ Feature Importance Chart Saved as 'feature_importance.png'")

# -------------------------------------------------------------
# 3. LIME Explanations (Local Interpretability) - Fixed
# -------------------------------------------------------------
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=list(X_train.columns),
    class_names=['No Churn', 'Churn'],
    mode='classification'
)

# .iloc[...].values பயன்படுத்தி 1D NumPy Array-ஆக அனுப்ப வேண்டும்
exp_churn = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
exp_churn.save_to_file('lime_churn_customer.html')

exp_no_churn = explainer.explain_instance(X_test.iloc[1].values, model.predict_proba)
exp_no_churn.save_to_file('lime_no_churn_customer.html')
print("✅ LIME Explanations Saved as HTML files!")

# -------------------------------------------------------------
# 4. Data Validation & Robust Prediction Pipeline
# -------------------------------------------------------------
def predict_new_customer(customer_data: dict, model, feature_names: list) -> float:
    # Missing Fields Check
    missing_fields = [field for field in feature_names if field not in customer_data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Data Type & Range Check
    for field, value in customer_data.items():
        if field in feature_names:
            if not isinstance(value, (int, float, np.number)):
                raise TypeError(f"Invalid data type for '{field}': expected numeric, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"Invalid value for '{field}': {value}. Value cannot be negative.")

    input_df = pd.DataFrame([customer_data])[feature_names]
    return float(model.predict_proba(input_df)[0][1])

# Testing Validation Pipeline
print("\n--- Testing Pipeline Validation ---")
# Valid Input
good_data = {'tenure': 12, 'MonthlyCharges': 75.0, 'TotalCharges': 900.0}
print("Valid Customer Churn Risk:", predict_new_customer(good_data, model, list(X.columns)))

# Bad Input 1 (Missing Field)
try:
    predict_new_customer({'tenure': 12}, model, list(X.columns))
except ValueError as e:
    print("Caught Expected Error (Missing Field):", e)

# Bad Input 2 (Negative Value)
try:
    predict_new_customer({'tenure': -5, 'MonthlyCharges': 75.0, 'TotalCharges': 900.0}, model, list(X.columns))
except ValueError as e:
    print("Caught Expected Error (Negative Value):", e)