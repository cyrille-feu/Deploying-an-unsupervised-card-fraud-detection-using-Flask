# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from utils.model_predict import model_predict
import pandas as pd
import numpy as np
import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")



@app.route('/predict', methods = ['GET','POST'])
def results():
    if request.method == 'POST':
        input_data = {
        "Amount":request.form.get("amount"),
        "Merchant": request.form.get("merchant"),
        "Location": request.form.get("location"),
        "TimeOfDay":request.form.get("timeOfDay"),
        "TransactionType":request.form.get("transactionType")
        }
        results = model_predict(input_data)
    else:
        results = {"Amount": "",
                "Merchant": "",
                "Location": "",
                "TimeOfDay": "",
                "TransactionType": "",
                "Predictions": ""
                }
    return render_template("results.html", results=results)



@app.route('/api/predict', methods=['GET','POST'])
def api_predict():
    try:
        # Get input data from the request
        input_data = request.get_json(force=True)

        results = model_predict(input_data)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
