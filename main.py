import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np


tickers = ['AAPL' ,'ABT','ADC','AEHR','ALB','ALGM','APD','BCRX','BLUE','C','CALX','CIEN','ENPH','ENVX','GOEV','GT'
            ,'HE','INTC','KHC','LEG','MBLY','MBRX','MDT','MMM','NEM','NIO','NKE','O','PARA','PEP','PFE','PLUG','PRO','PYPL',
            'RKT','SHLS','STT','TSLA','TTWO','WBD','GPRO','JNJ']

end_date = '2024-04-01'
start_date = '2023-01-01'


def prepare_data(tickers:list, steps_back:int)-> pd.DataFrame:
    new_data = []
    for ticker in tickers:
        df = yf.download(tickers=ticker, start=start_date, end=end_date)
        for i in range(df.shape[0]-steps_back):
            row = df['Close'].iloc[i:i+steps_back].values.ravel()
            row = row.tolist()
            row.append(df['Close'].iloc[i+steps_back])
            new_data.append(row)

        columns = ['3-da', '2-da', '1-da', 'now']
        new_df = pd.DataFrame(new_data, columns=columns)
    return new_df


# data = prepare_data(data, 5)
# Y = data[data.columns[-1]]
# X = data.drop(data.columns[-1], axis=1)
# 
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# 
# random_forest = RandomForestRegressor(n_estimators=300)
# random_forest.fit(X_train, Y_train)

# linear = LinearRegression()
# linear.fit(X_train, Y_train)

# mse_random_forest = mean_squared_error(Y_test, random_forest.predict(X_test))
# mse_linear = mean_squared_error(Y_test, linear.predict(X_test))
# print(f'Random Forest MSE = {mse_random_forest}')
# print(f'Linear Regressor MSE = {mse_linear}')


kf = KFold(n_splits=5)
df = prepare_data(tickers=tickers, steps_back=3)
Y = df[df.columns[-1]]
X = df.drop(df.columns[-1], axis=1)
mse = []
for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    model = LinearRegression()
    model.fit(X_train, Y_train)
    mse.append(mean_squared_error(Y_test, model.predict(X_test)))
print(np.mean(mse))


master_model = LinearRegression()
master_model.fit(X, Y)



df = prepare_data(tickers=['GOOG'], steps_back=3)

X = df.drop(df.columns[-1], axis=1).iloc[:3]    
for i in range(df.shape[0]-3):
    new_row = {'3-da': X['2-da'].iloc[-1], 
                '2-da': X['1-da'].iloc[-1],
               '1-da': master_model.predict(X.iloc[-1].values.reshape(1, -1))}
    new_row = pd.DataFrame(new_row)
    new_row.index = [i+3]
    X = pd.concat([X, new_row])
print(X)


sn.set()

plt.plot(df[df.columns[0]], label='Actual Data')
plt.plot(X['3-da'], label='Predicted Data')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend()
plt.show()