import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tabulate import tabulate

def give_non_advice(stonk):
  ticker = stonk
  start_date = '1901-01-01'
  end_date = '2023-05-05'
  data = yf.download(ticker, start=start_date, end=end_date)
  data['SMA10'] = data['Close'].rolling(window=10).mean()
  data['SMA50'] = data['Close'].rolling(window=50).mean()
  data['SMA200'] = data['Close'].rolling(window=200).mean()
  data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
  data.dropna(inplace=True)
  train_size = int(len(data) * 0.8)
  train_data = data[:train_size]
  test_data = data[train_size:]
  features = ['SMA10', 'SMA50', 'SMA200']
  model = LogisticRegression()
  model.fit(train_data[features], train_data['Target'])
  predictions = model.predict(test_data[features])
  accuracy = accuracy_score(test_data['Target'], predictions)
  next_day_features = data.iloc[-1][features].values.reshape(1, -1)
  next_day_prediction = model.predict(next_day_features)
  n_advice = 'N/A'
  if next_day_prediction == 1:
    n_advice = 'Buy'
  else:
    n_advice = 'Sell'
  row = [stonk, accuracy, n_advice]
  return row

my_stonks = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOG', 'BRK', 'NVDA', 'V', 'UNH']
my_rows = []
for stonk in my_stonks:
  row = give_non_advice(stonk)
  my_rows.append(row)
print(("\n")*3, ('🚀'*30), ("\n"), ('🚀'*30), ("\n"), ('🚀'*30), ("\n")*3)
print(tabulate(my_rows, headers=["Stonk", "Accuracy", "Non-Advice"]), ("\n")*3, ('🚀'*30), ("\n"), ('🚀'*30), ("\n"), ('🚀'*30))
