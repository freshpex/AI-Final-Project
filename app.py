from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the model and scalers
model = load_model('combined_stock_prediction_model.h5')
scalers = {company: joblib.load(f'scaler_{company}.pkl') for company in ['Apple', 'LG', 'Tesla', 'Google', 'Netflix']}

# Load combined dataset
df = pd.read_csv('path/to/combined_stock_data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    date_str = request.form['date']
    date = pd.to_datetime(date_str)

    # Validate date
    company_data = df[df['Company'] == company]
    if date not in company_data['Date'].values:
        available_dates = f"{company_data['Date'].min().date()} to {company_data['Date'].max().date()}"
        return f"Date out of range. Available dates are: {available_dates}"

    # Prepare data for prediction
    time_step = 60
    scaler = scalers[company]
    company_scaled_data = company_data.drop(['Company', 'Date'], axis=1).values
    scaled_values = scaler.transform(company_scaled_data)

    x_test = []
    for i in range(time_step, len(company_scaled_data)):
        x_test.append(scaled_values[i-time_step:i])

    x_test = np.array(x_test)
    y_pred = model.predict(x_test)
    predicted_price = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], company_scaled_data.shape[1]-1)), y_pred), axis=1))[:, -1]
    predicted_price_on_date = predicted_price[company_data['Date'].get_loc(date) - time_step]

    # Plotting the results
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(company_data['Date'], company_data['Close'], label='Actual Stock Price')
    ax.plot(company_data['Date'][time_step:], predicted_price, label='Predicted Stock Price')
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
    df = pd.read_csv(f'datasets/{company}.csv', index_col='Date', parse_dates=True)
    
    # Prepare data for prediction
    time_step = 60
    x_test = []
    for i in range(time_step, len(df)):
        x_test.append(df.iloc[i-time_step:i].values)

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Predict using the model
    y_pred = model.predict(x_test)
    predicted_price = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], df.shape[1]-1)), y_pred), axis=1))[:, -1]
    predicted_price_on_date = predicted_price[-1]

    # Adjust the predicted price based on the custom open price
    adjustment_factor = open_price / df['Open'].iloc[-1]
    adjusted_predicted_price_on_date = predicted_price_on_date * adjustment_factor

    return render_template('result.html', prediction_date=custom_date.date(), predicted_price=adjusted_predicted_price_on_date)

if __name__ == '__main__':
    app.run(debug=True)
