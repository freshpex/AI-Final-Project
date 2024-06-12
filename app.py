from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

app = Flask(__name__)

# Load pre-trained model
model = load_model('stock_prediction_model.h5')

# Load the default dataset
default_data = pd.read_csv('tesla.csv')
default_data["Date"] = pd.to_datetime(default_data["Date"])
default_data.set_index('Date', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
default_scaled_data = scaler.fit_transform(default_data[['Close']])

def create_dataset(data, time_step=1):
    X = []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), 0]
        X.append(a)
    return np.array(X)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    predictions = []
    
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        data = pd.read_csv(file)
        data["Date"] = pd.to_datetime(data["Date"])
        data.set_index('Date', inplace=True)
        data = data[['Close']]
        scaled_data = scaler.fit_transform(data.values)
        time_step = 60
        X = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        predictions = predictions.tolist()
    
    elif 'date' in request.form:
        date_str = request.form['date']
        input_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        
        if input_date in default_data.index:
            idx = default_data.index.get_loc(input_date)
            if idx >= 60:
                input_data = default_scaled_data[idx-60:idx]
                input_data = input_data.reshape(1, 60, 1)
                prediction = model.predict(input_data)
                prediction = scaler.inverse_transform(prediction)
                predictions.append(prediction[0][0])
            else:
                predictions.append("Not enough data to predict for this date.")
        else:
            predictions.append("Date is out of range of the dataset.")
    
    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
