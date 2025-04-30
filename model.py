import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ticker = "Stock_name"  #Change the Stock name to the Stock you want to predict 
data = yf.download(ticker, start="1960-01-01", end="2025-04-28")
print(data)

data = data.reset_index()
data['Days'] = np.arange(len(data))

x = data[['Days']]
y = data['Close']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 )
poly = PolynomialFeatures(degree=5)  
x_poly = poly.fit_transform(x_train)
model = LinearRegression()
model.fit(x_poly, y_train)

x_test_poly = poly.transform(x_test)
y_pred = model.predict(x_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

sorted_indices = np.argsort(x_test.squeeze().values)  
x_test_sorted = x_test.squeeze().values[sorted_indices]
y_pred_sorted = y_pred.squeeze()[sorted_indices]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual Price')  # actual points
plt.plot(x_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Predicted Line')  # smooth line
plt.title('AAPL Stock Price Prediction using Linear Regression')
plt.xlabel('Days')
plt.ylabel('Price USD')
plt.legend()
plt.show()
