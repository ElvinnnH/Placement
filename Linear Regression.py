import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
  
# Load data from Excel
excel_file = '/Users/elvinhuang/Desktop/Placement/Regression Analysis.xlsx'
sheets_to_process = ['S3GO Y', 'RCAP Y', 'HYGG Y']

# Define the dependent variable and the independent variables
dependent_var = 'Flow'
independent_vars = ['Return', 'CPI', 'Unemployment', 'Cash Rate', 'mid term bond']

for sheet_name in sheets_to_process:
    print(f"Processing {sheet_name}...")

    data = pd.read_excel(excel_file, sheet_name=sheet_name)
    data.set_index('Month', inplace=True)
    y = data[dependent_var]
    X = data[independent_vars]

    # Original Variables Regression
    X1 = sm.add_constant(X)
    model1 = sm.OLS(y, X1).fit()
    predictions1 = model1.get_prediction(X1)
    prediction_summary_frame1 = predictions1.summary_frame()
    predicted_mean1 = prediction_summary_frame1['mean']
    lower_bounds1 = prediction_summary_frame1['obs_ci_lower']
    upper_bounds1 = prediction_summary_frame1['obs_ci_upper']

    # Plotting for Original Model
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, y, label='Actual Data', color='blue')
    plt.plot(data.index, predicted_mean1, label='Predicted Data (Original Model)', color='red')
    plt.fill_between(data.index, lower_bounds1, upper_bounds1, color='gray', alpha=0.3, label='Prediction Interval (Original Model)')
    plt.title(f'Actual vs Predicted (Original Model) for {sheet_name}')
    plt.xlabel('Index')
    plt.ylabel(dependent_var)
    plt.legend()
    plt.show()
    # Print Model Summaries
    print("Model with Original Variables:")
    print(model1.summary())
    # Print the regression equation
    print("Regression Equation (Original Model):")
    print(f"{dependent_var} = {' + '.join([f'{coef:.4f}*{var}' for coef, var in zip(model1.params[1:], independent_vars)])} + {model1.params[0]:.4f}")

    # Lagged Variables Regression
    X_lagged = sm.add_constant(X.shift(1).fillna(0))  # Create lagged DataFrame
    model2 = sm.OLS(y, X_lagged).fit()
    predictions2 = model2.get_prediction(X_lagged)
    prediction_summary_frame2 = predictions2.summary_frame()
    predicted_mean2 = prediction_summary_frame2['mean']
    lower_bounds2 = prediction_summary_frame2['obs_ci_lower']
    upper_bounds2 = prediction_summary_frame2['obs_ci_upper']

    # Plotting for Lagged Model
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, y, label='Actual Data', color='blue')
    plt.plot(data.index, predicted_mean2, label='Predicted Data (Lagged Model)', color='green')
    plt.fill_between(data.index, lower_bounds2, upper_bounds2, color='yellow', alpha=0.3, label='Prediction Interval (Lagged Model)')
    plt.title(f'Actual vs Predicted (Lagged Model) for {sheet_name}')
    plt.xlabel('Index')
    plt.ylabel(dependent_var)
    plt.legend()
    plt.show()
    # Print Model Summaries
    print("\nModel with Lagged Variables:")
    print(model2.summary())
    # Print the regression equation
    print("Regression Equation (Lagged Model):")
    print(f"{dependent_var} = {' + '.join([f'{coef:.4f}*{var}_lag' for coef, var in zip(model2.params[1:], independent_vars)])} + {model2.params[0]:.4f}")
    # Print original and predicted data for Lagged Model
    print("\nOriginal and Predicted Data (Lagged Model):")
    print(pd.concat([y, predicted_mean2], axis=1))

    print("\n" + "="*50 + "\n")

# Evaluate and choose the better model based on R-squared, AIC, BIC for each sheet

