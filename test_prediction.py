import joblib
import pandas as pd

# 1. Load the Saved Pipeline (.pkl)
pipeline = joblib.load("churn_pipeline.pkl")

# 2. Brand-new Customer Data
new_customer = pd.DataFrame(
    [{"tenure": 2, "MonthlyCharges": 70.35, "TotalCharges": 140.70}]
)

# 3. Predict without retraining
prediction = pipeline.predict(new_customer)
probability = pipeline.predict_proba(new_customer)[0][1]

print("--- Prediction Results ---")
print(f"Churn Prediction: {prediction[0]}")
print(f"Churn Probability: {probability:.2%}")