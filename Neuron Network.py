import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
import random
import os

# Set a seed value
seed_value= 42
  
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

# Load data from Excel
excel_file = '/Users/elvinhuang/Desktop/Placement/Regression Analysis.xlsx'
sheet_name = 'HYGG Y'
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Assuming the names of your columns
feature_columns = ['Return', 'CPI', 'Cash Rate', 'Unemployment', 'mid term bond']
target_column = 'Flow'

X = df[feature_columns].values
y = df[target_column].values

# Initialize LOOCV
loo = LeaveOneOut()

# Lists to store actual values and predictions
actual_values = []
predictions = []
training_losses = []

# Loop over each cross-validation split
for train_index, test_index in loo.split(X):
    # Split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Create a neural network model
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model and store the history
    history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

    # Print the final training loss for this iteration
    final_loss = history.history['loss'][-1]  # Get the last loss value
    print(f"Final training loss for this iteration: {final_loss}")

    # Record the training loss
    training_losses.append(history.history['loss'])

    # Make a prediction and inverse transform it
    pred_scaled = model.predict(X_test).flatten()
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    predictions.extend(pred)

    # Inverse transform the actual test values
    actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    actual_values.extend(actual)

    print(f"Predictions: {pred}")
    print(f"Actual values: {actual}")

# Calculate and print the mean MSE
mean_mse = np.mean([mse[-1] for mse in training_losses])  # Use the last MSE of each fold
print(f"Mean MSE from LOOCV: {mean_mse}")

# Plotting the mean training loss across all LOOCV iterations
mean_training_loss = np.mean(training_losses, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(mean_training_loss, label='Mean Training Loss')
plt.title('Mean Training Loss Across LOOCV Iterations')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.show()

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(actual_values, label='Actual Values')
plt.plot(predictions, label='Predictions', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Data Point Index')
plt.ylabel('Target Value')
plt.legend()
plt.show()

