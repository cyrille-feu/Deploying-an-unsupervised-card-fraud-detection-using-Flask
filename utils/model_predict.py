import pandas as pd
import joblib
from flask import jsonify

def model_predict(input_data):
    # Load the model and features
    model = joblib.load('models/isolation_forest_model.joblib')
    features = joblib.load('models/model_features.joblib')


    # Preprocess input data
    # Fill missing values with mean for 'Amount' column, encode categorical variables using one-hot encoding
    input_df = pd.DataFrame([input_data])
    input_df['Amount'] = input_df['Amount'].astype(float)
    input_df['Amount'].fillna(input_df['Amount'].mean(), inplace=True)
    input_df = pd.get_dummies(input_df, columns=['Merchant', 'Location', 'TimeOfDay', 'TransactionType'], drop_first=True)

    # Ensure all columns present during training are also present in the input data
    missing_columns = set(features) - set(input_df.columns)

    # Add missing columns with a value of 0
    for col in missing_columns:
        input_df[col] = 0

    # Reorder columns to match the order during training
    input_df = input_df[features]

    # Make predictions
    predictions = model.predict(input_df)

    updated_prediction = [ "normal" if i == 1  else "abnormal" for i in predictions.tolist()]

    result_dict = {"Amount": input_data["Amount"],
                    "Merchant": input_data["Merchant"],
                    "Location": input_data["Location"],
                    "TimeOfDay": input_data["TimeOfDay"],
                    "TransactionType": input_data["TransactionType"],
                    "Predictions": updated_prediction[0]}

    return result_dict


if __name__ == "__main__":

    input_data = {
        "Amount":800,
        "Merchant": "A",
        "Location": "Local",
        "TimeOfDay":"Morning",
        "TransactionType":"Withdrawal"
    }

    model_predict(input_data)