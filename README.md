Iris Flower Classification API

1. Project Overview

The Iris Flower Classification API is a machine learning project that predicts the species of an Iris flower based on four measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The project uses the well-known Iris dataset available through "scikit-learn".

The API will accept the four flower measurements as input and return the predicted Iris species as the output.

The main goal of Task 1 is to understand the machine learning problem, define the API contract, and plan the project architecture before implementing the Python code.

---

2. Problem Statement

The objective of this project is to classify an Iris flower into one of three species based on its physical measurements.

The three possible species are:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

This is a Supervised Machine Learning Classification problem because the model predicts one category from a fixed set of classes.

---

3. Dataset

The project will use the Iris dataset provided by scikit-learn.

Input Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target Variable

The target variable is the Iris flower species.

The dataset can be accessed using the built-in "load_iris()" function from scikit-learn.

Dataset Source: "sklearn.datasets.load_iris"

---

4. API Contract

The API will accept four numerical flower measurements as input.

Input

The API request will contain:

Parameter| Data Type| Description
"sepal_length"| float| Sepal length of the flower
"sepal_width"| float| Sepal width of the flower
"petal_length"| float| Petal length of the flower
"petal_width"| float| Petal width of the flower

Example Input

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

Output

The API will return the predicted Iris species.

Example Output

{
  "predicted_species": "Iris-setosa"
}

The API will validate the input before sending the data to the machine learning model.

---

5. Request → Validation → Model → Response Flow

The planned request flow is:

Client / User
      ↓
API Request
      ↓
Input Validation
      ↓
Machine Learning Model
      ↓
Prediction
      ↓
API Response

Flow Explanation

1. The client or user sends the four flower measurements to the API.
2. The API receives the request.
3. The API validates the input values and data types.
4. Valid input is passed to the trained machine learning model.
5. The model predicts the Iris flower species.
6. The API returns the prediction to the client.

---

6. Planned Machine Learning Model

The initial planned machine learning model is Logistic Regression.

Logistic Regression is selected because:

- It is simple and beginner-friendly.
- It is suitable for classification problems.
- It works well with the Iris dataset.
- It is easy to understand and implement.

The model will be trained using the Iris dataset and will learn the relationship between the four flower measurements and the flower species.

The actual model training and API implementation will be completed in later tasks.

---

7. High-Level Architecture

flowchart TD
    A[Client / User] --> B[API Request]
    B --> C[Input Validation]
    C --> D[ML Model]
    D --> E[Logistic Regression]
    E --> F[Prediction]
    F --> G[API Response]

Architecture Explanation

- Client / User: Sends flower measurements.
- API Request: Receives the input data.
- Input Validation: Checks whether the input values are valid.
- ML Model: Processes the validated measurements.
- Logistic Regression: Predicts the flower species.
- Prediction: Generates the predicted class.
- API Response: Sends the prediction back to the client.

---

8. Planned Project Structure

The actual project structure will be developed in Task 2.

The planned structure is:

iris-flower-classification-api/
│
├── README.md
├── requirements.txt
│
├── app/
│   ├── main.py
│   ├── model.py
│   └── schemas.py
│
├── model/
│   └── trained_model.pkl
│
└── tests/
    └── test_api.py

This structure is only a plan for Task 1. The working Python files will be created in the next task.

---

9. Minimum Viable Product (MVP)

The minimum version of this project should be able to:

1. Accept four Iris flower measurements.
2. Validate the input.
3. Pass the input to the trained machine learning model.
4. Predict the flower species.
5. Return the prediction through an API response.

Additional features can be considered later if required.

---

10. Task 1 Deliverables

The following items will be completed for Task 1:

- [x] Iris dataset selected
- [x] Machine learning problem identified
- [x] API input and output contract defined
- [x] Request → Validation → Model → Response flow planned
- [x] High-level architecture documented
- [x] Planned machine learning model identified
- [x] GitHub repository created
- [x] "README.md" added and documented

---

11. Scope of Task 1

Task 1 focuses only on understanding and planning the project.

No complete Python implementation is required at this stage.

The following activities will be completed in later tasks:

- Python environment setup
- Project folder structure
- Dependency installation
- Machine learning model implementation
- Model training
- API development
- Input validation
- API testing

---

12. Next Step

After completing Task 1, the project will move to Task 2.

Task 2 will convert this plan into an actual Python project by:

1. Creating the project folder structure.
2. Setting up the Python environment.
3. Installing the required dependencies.
4. Creating the initial Python files.
5. Preparing the project for machine learning model development and API implementation.

---

Conclusion

The Iris Flower Classification API will provide a simple machine learning-based API for predicting Iris flower species from four physical measurements.

Task 1 establishes the project requirements, API contract, architecture, dataset, and machine learning approach before moving to the actual implementation in Task 2.
