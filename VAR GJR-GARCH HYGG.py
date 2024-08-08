import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from arch import arch_model
import numpy as np

# Load your data from Excel
excel_file = '/Users/elvinhuang/Desktop/Placement/Regression Analysis.xlsx'
sheet_name = 'HYGG Y'

df = pd.read_excel(excel_file, sheet_name=sheet_name)

df['Month'] = pd.to_datetime(df['Month'])

# Extract the columns for the two time series
ts1 = df['DFlow']
ts2 = df['Return']

# Combine the two time series into one DataFrame
data = pd.concat([ts1, ts2], axis=1)
data.columns = ['DFlow', 'Return']

# Clean the data by removing rows with missing values
data.dropna(inplace=True)

# Determine the maximum lag order for the VAR model
max_p = 3

# Initialize variables to store the best AIC value and lag order
best_aic = float('inf')
best_order = (0, 0)

# Iterate over different lag orders and fit VAR models
for p in range(1, max_p + 1):
    model = VAR(data)
    results = model.fit(p)

    # Calculate the AIC
    aic = results.aic

    # Update the best AIC and lag order if a lower AIC is found
    if aic < best_aic:
        best_aic = aic
        best_order = (p)

# Fit the VAR model with the best lag order
best_p = best_order
final_model = VAR(data)
final_results = final_model.fit(best_p)

# Print the summary of the final VAR model
print(final_results.summary())

# Get and print the residuals
residuals = final_results.resid['DFlow']
print(residuals)

# Forecast future values
forecast_steps = 4  # Adjust this for the number of time steps you want to forecast
forecast = final_results.forecast(data.values[-best_p:], steps=forecast_steps)

# Create a time index for the forecast
last_date = df['Month'].max()
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='M')

# Combine actual and forecasted data for seamless plotting
combined_dates = df['Month'].tolist() + forecast_index.tolist()
combined_flow = df['DFlow'].tolist() + forecast[:, 0].tolist()

# Reverse the differencing for the forecast
forecast_original_flow = [df['Flow'].iloc[-1] + sum(forecast[:i+1, 0]) for i in range(forecast_steps)]

# Combine actual and forecasted data for seamless plotting
combined_flow_original = df['Flow'].tolist() + forecast_original_flow  # Assuming 'Flow' is the original series

# Plot the combined data for original flow
plt.figure(figsize=(12, 6))
plt.plot(combined_dates, combined_flow_original, label='Original Flow Forecast', color='red')
plt.plot(df['Month'], df['Flow'], label='Actual Original Flow', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('VAR Model Forecast for Original Flow')
plt.legend()
plt.show()

# Extract the coefficients and intercept for the linear function
coefficients = final_results.coefs
intercept = final_results.intercept

# Create a linear equation for 'Flow' based on the VAR model
linear_equation = f'Flow = {intercept[0]:.2f}'

for i in range(best_p):
    for j, col in enumerate(data.columns):
        variable_name = col if i == 0 else f'{col}(t-{i})'
        linear_equation += f' + {coefficients[i][0][j]:.2f} * {col}(t-{i+1})'

print("Linear Equation for Diff_Flow:")
print(linear_equation)

residuals_scaled = residuals
# Fit a GJR-GARCH model to these residuals
gjr_garch_model = arch_model(residuals_scaled, vol='Garch', p=1, o=1, q=1)
gjr_garch_fit = gjr_garch_model.fit()

# Print out the summary of the GJR-GARCH model
print(gjr_garch_fit.summary())
# Extract the estimated parameters
params = gjr_garch_fit.params
omega, alpha, gamma, beta = params['omega'], params['alpha[1]'], params['gamma[1]'], params['beta[1]']

# Print the variance equation
print(f"Conditional Variance Equation: σ² = {omega:.4f} + {alpha:.4f} * ε²(t-1) + {gamma:.4f} * ε²(t-1) * I[ε(t-1) < 0] + {beta:.4f} * σ²(t-1)")

# Forecasting the next 10 periods
var_forecasts = gjr_garch_fit.forecast(horizon=forecast_steps)
forecasted_volatility = var_forecasts.variance.values[-1, :]  # Last row contains the forecast
forecasted_std_dev = np.sqrt(forecasted_volatility)

# Create confidence intervals
upper_bound = forecast[:, 0] + 1.96 * forecasted_std_dev
lower_bound = forecast[:, 0] - 1.96 * forecasted_std_dev
print(forecast[:, 0])
print(upper_bound)
print(lower_bound)

# Plotting
# Combine actual and forecasted data for seamless plotting
start_date_for_fitted = df['Month'].iloc[best_p:]
fitted_values = final_results.fittedvalues['DFlow']
combined_dates =start_date_for_fitted.tolist() + forecast_index.tolist()
combined_flow = fitted_values.tolist() + forecast[:, 0].tolist()
# Plot the combined data
plt.figure(figsize=(12, 6))
plt.plot(combined_dates, combined_flow, label='Flow Forecast', color='red')
# Plot Fitted Values from VAR Model


# Plot Fitted Values from VAR Model
plt.plot(start_date_for_fitted, fitted_values, label='VAR Model Fitted Values', color='green')
plt.plot(df['Month'], df['DFlow'], label='Actual DFlow', color='blue')
plt.fill_between(forecast_index, lower_bound, upper_bound, color='pink', alpha=1, label='Prediction Interval')
plt.plot(forecast_index, forecast[:, 0], color='red')
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Combined VAR and GARCH Forecast')
plt.legend()
plt.show()
print("Forecasted Standard Deviations:")
print(forecasted_std_dev)


