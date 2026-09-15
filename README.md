# Iris Flower Classification MLOps API

A production-ready, containerized Machine Learning API built with FastAPI, Docker, Prometheus, and Locust. This project demonstrates an end-to-end MLOps lifecycle from model training and validation to deployment, containerization, monitoring, and automated testing.

---

## 📋 1. Project Overview
The Iris Flower Classification API is a machine learning project that predicts the species of an Iris flower based on four physical measurements:
- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

The project uses the well-known Iris dataset from `scikit-learn` and a **Logistic Regression** model for classification into three species: `Iris-setosa`, `Iris-versicolor`, and `Iris-virginica`.

---

## 🏗️ 2. High-Level Architecture & Flow
```text
Client / User --> API Request --> Input Validation --> ML Model (Logistic Regression) --> Prediction --> API Response