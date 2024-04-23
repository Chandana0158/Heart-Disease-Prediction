from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Define column names
columns = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'cp_1', 'cp_2', 'cp_3', 'restecg_1', 'restecg_2', 'thal_1', 'thal_2', 'thal_3']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = []
    input_values = {}  # Dictionary to store input values
    for col in columns:
        value = float(request.form[col])
        features.append(value)
        input_values[col] = value  # Store input value in dictionary
    data = pd.DataFrame([features], columns=columns)
    prediction = model.predict(data)
    if prediction[0] == 1:
        result = 'Heart Disease Detected'
    else:
        result = 'No Heart Disease Detected'
    return render_template('result.html', input_values=input_values, prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)

