import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats

file_path = 'monthly_fire_distributions.csv'
fire_data = pd.read_csv(file_path)

fire_data['month'] = pd.to_datetime(fire_data['month'])
fire_data.set_index('month', inplace=True)

fire_data['number_of_fire'].fillna(method='ffill', inplace=True)

result = seasonal_decompose(fire_data['number_of_fire'], model='additive')
result.plot()
plt.suptitle('Time Series Decomposition', fontsize=16)
plt.show()

adf_test = adfuller(fire_data['number_of_fire'])
print(f'ADF Statistic: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')
for key, value in adf_test[4].items():
    print(f'Critical Value {key}: {value}')

plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_acf(fire_data['number_of_fire'], lags=20, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')

plt.subplot(122)
plot_pacf(fire_data['number_of_fire'], lags=20, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

def find_best_arima(data, p_values, d_values, q_values):
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(data, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except:
                    continue
    return best_order, best_model

p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]
best_order, best_model = find_best_arima(fire_data['number_of_fire'], p_values, d_values, q_values)
print(f'Best ARIMA order: {best_order}')

forecast_steps = 60
forecast, stderr, conf_int = best_model.forecast(steps=forecast_steps, alpha=0.05)
forecast_index = pd.date_range(start=fire_data.index[-1], periods=forecast_steps + 1, freq='MS')[1:]
forecast_series = pd.Series(forecast, index=forecast_index)

plt.figure(figsize=(10, 5))
plt.plot(fire_data.index, fire_data['number_of_fire'], label='Observed')
plt.plot(forecast_series.index, forecast_series, label='Forecast', color='red')
plt.fill_between(forecast_series.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.xlabel('Year')
plt.ylabel('Number of Fires')
plt.title('Observed vs Predicted Fire Data (2020-2025)')
plt.legend()
plt.show()

train_size = int(len(fire_data) * 0.8)
train_data, test_data = fire_data['number_of_fire'][:train_size], fire_data['number_of_fire'][train_size:]

train_model = ARIMA(train_data, order=best_order).fit()
test_forecast = train_model.forecast(steps=len(test_data))

mse = mean_squared_error(test_data, test_forecast)
mae = mean_absolute_error(test_data, test_forecast)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

future_steps = 60
future_forecast, stderr, conf_int_future = best_model.forecast(steps=future_steps, alpha=0.05)
future_forecast_index = pd.date_range(start='2023-01-01', periods=future_steps, freq='MS')
future_forecast_series = pd.Series(future_forecast, index=future_forecast_index)

forecast_df = pd.DataFrame({'month': future_forecast_series.index, 'number_of_fire': future_forecast_series.values})
forecast_df.to_csv('fire_forecast_2023_2027.csv', index=False)

plt.figure(figsize=(10, 5))
plt.plot(future_forecast_series.index, future_forecast_series, label='Forecast (2023-2027)', color='blue')
plt.fill_between(future_forecast_series.index, conf_int_future[:, 0], conf_int_future[:, 1], color='lightblue', alpha=0.3)
plt.xlabel('Year')
plt.ylabel('Number of Fires')
plt.title('Predicted Fire Data (2023-2027) with Confidence Interval')
plt.legend()
plt.show()

residuals = best_model.resid

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(residuals)
plt.title('Residuals Plot')
plt.subplot(212)
sns.histplot(residuals, kde=True, bins=20)
plt.title('Residuals Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()

ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(f'Ljung-Box Test p-value: {ljung_box_result["lb_pvalue"].values[0]}')
