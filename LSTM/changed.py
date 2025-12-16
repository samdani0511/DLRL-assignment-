import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.utils import plot_model
import kagglehub

# --- Download dataset using kagglehub ---
dataset_path = kagglehub.dataset_download("andreazzini/international-airline-passengers")
csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
file_path = os.path.join(dataset_path, csv_files[0])
print("Using dataset:", file_path)

# --- Load dataset ---
data = pd.read_csv(file_path)
dataset = data.iloc[:, 1].values.astype('float32').reshape(-1, 1)

# --- Plot raw data ---
plt.figure(figsize=(10,4))
plt.plot(dataset, label='Passengers')
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.title("International Airline Passengers")
plt.grid(True)
plt.legend()
plt.show()

# --- Scale dataset ---
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# --- Split into train and test sets ---
train_size = int(len(dataset_scaled) * 0.75)
train, test = dataset_scaled[:train_size], dataset_scaled[train_size:]
print(f"Train size: {len(train)}, Test size: {len(test)}")

# --- Helper function to create sequences ---
def create_sequences(data, time_steps=10):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 0])
        Y.append(data[i + time_steps, 0])
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, Y

time_steps = 10
trainX, trainY = create_sequences(train, time_steps)
testX, testY = create_sequences(test, time_steps)

# --- Build LSTM model ---
model = Sequential([
    LSTM(50, input_shape=(1, time_steps)),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

# --- Train model ---
history = model.fit(
    trainX, trainY, 
    epochs=100, 
    batch_size=1, 
    validation_split=0.1, 
    verbose=2
)

# --- Model summary & plot ---
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# --- Predictions ---
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# --- Inverse transform to original scale ---
trainPredict = scaler.inverse_transform(trainPredict)
trainY_inv = scaler.inverse_transform(trainY.reshape(-1,1))
testPredict = scaler.inverse_transform(testPredict)
testY_inv = scaler.inverse_transform(testY.reshape(-1,1))

# --- Calculate RMSE ---
trainScore = math.sqrt(mean_squared_error(trainY_inv, trainPredict))
testScore = math.sqrt(mean_squared_error(testY_inv, testPredict))
print(f'Train RMSE: {trainScore:.2f}')
print(f'Test RMSE: {testScore:.2f}')

# --- Plot predictions ---
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_steps:len(trainPredict)+time_steps, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_steps*1):len(dataset), :] = testPredict

plt.figure(figsize=(10,4))
plt.plot(dataset, label="Actual")
plt.plot(trainPredictPlot, label="Train Prediction")
plt.plot(testPredictPlot, label="Test Prediction")
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.title("Airline Passengers Prediction with LSTM")
plt.grid(True)
plt.legend()
plt.show()
