import joblib
import numpy as np

# Load saved model
model = joblib.load("ml/saved_model/model.joblib")

# Sample input: [sepal length, sepal width, petal length, petal width]
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])

# Run prediction
prediction = model.predict(sample_data)
print(f"Predicted Class ID: {prediction[0]}")