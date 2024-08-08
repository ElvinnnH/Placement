import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data from Excel
excel_file = '/Users/elvinhuang/Desktop/Placement/Regression Analysis.xlsx'
sheets_to_process = ['S3GO Lag', 'RCAP Lag', 'HYGG Lag']

# Define the dependent variable and the independent variables
dependent_var = 'Flow'
independent_vars = ['Return', 'CPI', 'Unemployment', 'Cash Rate', 'mid term bond']

for sheet_name in sheets_to_process:
    print(f"Processing {sheet_name}...")

    # Read data from Excel sheet
    data = pd.read_excel(excel_file, sheet_name=sheet_name)
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace=True)

    # Check the number of data points
    num_data_points = len(data)
    print(f"Number of data points in {sheet_name}: {num_data_points}")

    # Ensure there are at least 2 data points to perform regression
    if num_data_points < 2:
        print(f"Not enough data points in {sheet_name} to perform regression.")
        continue

    # Separate the data into training and prediction sets
    train_data = data.iloc[:-1]
    predict_data = data.iloc[-1]

    # Separate dependent and independent variables
    y_train = train_data[dependent_var]
    X_train = train_data[independent_vars]

    # Fit the linear regression model
    X_train = sm.add_constant(X_train)  # Add a constant term for the intercept
    model = sm.OLS(y_train, X_train).fit()

    # Prepare the last data point for prediction
    predict_row = pd.DataFrame(predict_data[independent_vars]).T
    X_predict = sm.add_constant(predict_row, has_constant='add')  # Add a constant

    # Ensure that X_predict has the correct number of columns
    if X_predict.shape[1] != model.params.size:
        raise ValueError("The number of columns in X_predict does not match the model's parameters.")

    # Predict using the model
    y_predict = model.predict(X_predict)

    # Compute prediction interval for the future point
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    pred_std, iv_l, iv_u = wls_prediction_std(model, exog=X_predict, alpha=0.05)
    iv_l_future = iv_l[0]
    iv_u_future = iv_u[0]

    # Plotting the actual data and the regression line with prediction interval
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, y_train, label='Actual Data', linestyle='-', marker='o', color='blue')
    plt.plot(train_data.index, model.predict(X_train), label='Regression Line', linestyle='-', marker='o', color='red')

    # Existing prediction intervals
    prediction = model.get_prediction(X_train)
    iv_l_train = prediction.summary_frame()['obs_ci_lower']
    iv_u_train = prediction.summary_frame()['obs_ci_upper']
    plt.fill_between(train_data.index, iv_l_train, iv_u_train, color='red', alpha=0.1, label='Prediction Interval')

    # Add the predicted future point and its interval
    plt.scatter(predict_data.name, y_predict, label='Future Prediction', color='green')
    plt.errorbar(predict_data.name, y_predict.iloc[0], yerr=[[y_predict.iloc[0] - iv_l_future], [iv_u_future - y_predict.iloc[0]]], fmt='o', color='green', label='Future Prediction Interval')

    plt.title(f'Actual Data vs Linear Regression Prediction for {sheet_name}')
    plt.xlabel('Date')
    plt.ylabel(dependent_var)
    plt.legend()
    plt.show()

    # Print Model Summaries
    print("\nModel with Lagged Variables:")
    print(model.summary())

    print("\n" + "="*50 + "\n")
