from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
flower_names = ['setosa', 'versicolor', 'virginica']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred_idx = model.predict(features)[0]
        prediction = flower_names[pred_idx]
        
    return render_template('index.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    features = np.array([[
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]])
    pred_idx = model.predict(features)[0]
    return jsonify({"prediction": flower_names[pred_idx]})

if __name__ == '__main__':
    app.run(debug=True)