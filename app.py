from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
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
    'tesla': 'datasets/tesla.csv',
    'apple': 'datasets/apple.csv',
    'lgtelevision': 'datasets/lgtelevision.csv',
    'netflix': 'datasets/netflix.csv',
    'google': 'datasets/google.csv'
}

def prepare_data_for_prediction(data, time_step=60):
    scaled_data = scaler.transform(data[['Close']])
    X = []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
    return np.array(X).reshape(-1, time_step, 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    date_str = request.form['date']
    date = pd.to_datetime(date_str)

    # Load the appropriate dataset
    df = pd.read_csv(datasets[company], index_col='Date', parse_dates=True)
    
    if date not in df.index:
        available_dates = f"{df.index.min().date()} to {df.index.max().date()}"
        return f"Date out of range. Available dates are: {available_dates}"

    # Prepare data for prediction
    time_step = 60
    X_test = prepare_data_for_prediction(df)

    y_pred = model.predict(X_test)
    predicted_price = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], df.shape[1]-1)), y_pred), axis=1))[:, -1]
    predicted_price_on_date = predicted_price[df.index.get_loc(date) - time_step]

    # Plotting the results
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index[time_step:], predicted_price, label='Predicted Stock Price')
    ax.axvline(x=date, color='r', linestyle='--', label=f'Prediction Date: {date.date()}')
    ax.set_title('Stock Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('result.html', plot_url=plot_url, prediction_date=date.date(), predicted_price=predicted_price_on_date)

@app.route('/predict_custom', methods=['POST'])
def predict_custom():
    company = request.form['company_custom']
    open_price = float(request.form['open_price'])
    custom_date_str = request.form['custom_date']
    custom_date = pd.to_datetime(custom_date_str)

    # Load the appropriate dataset
    df = pd.read_csv(datasets[company], index_col='Date', parse_dates=True)
    
    # Prepare data for prediction
    time_step = 60
    X_test = prepare_data_for_prediction(df)

    # Predict using the model
    y_pred = model.predict(X_test)
    predicted_price = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], df.shape[1]-1)), y_pred), axis=1))[:, -1]
    predicted_price_on_date = predicted_price[-1]

    # Adjust the predicted price based on the custom open price
    adjustment_factor = open_price / df['Open'].iloc[-1]
    adjusted_predicted_price_on_date = predicted_price_on_date * adjustment_factor

    return render_template('result.html', prediction_date=custom_date.date(), predicted_price=adjusted_predicted_price_on_date)

if __name__ == '__main__':
    app.run(debug=True)
