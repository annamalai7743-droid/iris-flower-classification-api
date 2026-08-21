import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create Pipeline (Preprocessing + Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# 4. Train
pipeline.fit(X_train, y_train)

# 5. Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save Model
joblib.dump(pipeline, "ml/saved_model/model.joblib")
print("Saved pipeline model to ml/saved_model/model.joblib successfully!")