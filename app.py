from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the model and scaler
model = load_model('models/stock_prediction_model.h5')
scaler = joblib.load('models/scaler.pkl')

# List of datasets
datasets = {
    'Tesla': 'datasets/tesla.csv',
    'Apple': 'datasets/apple.csv',
    'LG': 'datasets/lgtelevision.csv',
    'Netflix': 'datasets/netflix.csv',
    'Google': 'datasets/google.csv'
}

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        X.append(a)
        y.append(data[i + time_step, 3])  # 3 is the index for 'Close' column
    return np.array(X), np.array(y)

@app.route('/')
def index():
    return render_template('index.html', datasets=datasets.keys())

@app.route('/predict', methods=['POST'])
def predict():
    dataset_name = request.form['dataset']
    file_path = datasets[dataset_name]

    # Load the dataset
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Normalize the data
    scaled_data = scaler.transform(df)

    # Get user input for prediction date
    prediction_date = request.form['date']
    prediction_date = pd.to_datetime(prediction_date)

    if prediction_date not in df.index:
        available_dates = f"{df.index.min().date()} to {df.index.max().date()}"
        return f"Date out of range. Available dates are: {available_dates}"

    time_step = 60
    x_test = []
    for i in range(time_step, len(df)):
        x_test.append(scaled_data[i-time_step:i])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    y_pred = model.predict(x_test)
    predicted_price = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], scaled_data.shape[1]-1)), y_pred), axis=1))[:, -1]

    predicted_price_on_date = predicted_price[df.index.get_loc(prediction_date)]

    # Plotting the results
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df['Close'], label='Actual Stock Price')
    ax.plot(df.index[time_step:], predicted_price, label='Predicted Stock Price')
    ax.axvline(x=prediction_date, color='r', linestyle='--', label=f'Prediction Date: {prediction_date.date()}')
    ax.set_title('Stock Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('result.html', plot_url=plot_url, prediction_date=prediction_date.date(), predicted_price=predicted_price_on_date)

if __name__ == '__main__':
    app.run(debug=True)
