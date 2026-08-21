import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model as model.pkl
with open('ml/saved_model/model.pkl') as f:
    pickle.dump(model, f)

print("model.pkl created successfully!")