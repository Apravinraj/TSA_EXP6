### Developed By : Pravin Raj A
### Register No. : 212222240079
### Date : 

# Ex.No: 6               HOLT WINTERS METHOD


### AIM:

To create and implement Holt Winter's Method Model using python for AirTemp (C) in Water Quality.



### ALGORITHM:

1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'Date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them
7. You calculate the mean and standard deviation of the AirTemp (C) in Water Quality dataset, then fit a Holt-Winters model to the entire dataset and make future predictions
8. You plot the original sales data and the predictions

### PROGRAM:
```

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd


# Load the dataset
data = pd.read_csv('/content/BTC-USD(1).csv', index_col='Date', parse_dates=True)

# **Change:** Use 'Close' column instead of 'AirTemp (C)'
data = data['Close'].resample('MS').mean()  

scaler = MinMaxScaler()
data_scaled = pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index)

train_data = data_scaled[:int(len(data_scaled) * 0.8)]
test_data = data_scaled[int(len(data_scaled) * 0.8):]

fitted_model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='add', seasonal_periods=12
).fit()


# Forecast and evaluate
test_predictions_add = fitted_model_add.forecast(len(test_data))
# Ensure fillna is applied to the entire Series to handle NaNs, not just the forecast
test_predictions_add = test_predictions_add.fillna(method='ffill')
# Convert the Series to numeric, coercing NaNs to NaNs
test_predictions_add = pd.to_numeric(test_predictions_add, errors='coerce')


# Plot predictions
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='TRAIN', color='red')
plt.plot(test_data, label='TEST', color='yellow')
plt.plot(test_predictions_add, label='PREDICTION', color='black')
plt.title('Train, Test, and Additive Holt-Winters Predictions')
plt.legend(loc='best')
plt.show()

final_model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12).fit()

forecast_predictions = final_model.forecast(steps=12)

# **Change:** Use 'Close' for plotting as well
data.plot(figsize=(12, 8), legend=True, label='Close')  
forecast_predictions.plot(legend=True, label='Forecasted Close')  
plt.title('Bitcoin Close Price Forecast')  
plt.show()

```



### OUTPUT:

#### TEST PREDICTION

![download](https://github.com/user-attachments/assets/bf788c93-2ca0-405e-afc6-a0e79c6174ed)



#### FINAL PREDICTION

![download](https://github.com/user-attachments/assets/5fe2019b-6409-40aa-94b8-40c1d8e612c8)

### RESULT:

#### Thus the program run successfully based on the Holt Winters Method model.
