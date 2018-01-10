from pandas import *
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm
import os

from data import data_dir

stock_path = os.path.join(data_dir, 'stock_prices.csv')
dji_path = os.path.join(data_dir, 'DJI.csv')
prices_long = read_csv(stock_path, index_col=[0,1], parse_dates = True, squeeze=True)
dji_all = read_csv(dji_path, index_col = 0, parse_dates = True)

calc_returns = lambda x: np.log(x/x.shift(1))[1:]
scale = lambda x: (x - x.mean()) / x.std()

def OLSreg(y, Xmat):
    return sm.OLS(y, sm.add_constant(Xmat, prepend=True)).fit()

print(prices_long.head())
print(dji_all.head())

prices = prices_long.unstack()
# print(prices)
dates = list(prices.index)
dates.remove(dt.datetime(2002,2,1))
prices = prices.drop('DDR', axis = 1)
prices = prices.drop([dt.datetime(2002,2,1)])
print(prices.shape)
print(prices.head())

returns = prices.apply(calc_returns)

dji_all = dji_all.sort_index()
dji = dji_all['Close'].reindex(index = dates)
dji_ret = calc_returns(dji)

def make_pca_index(data, scale_data = True):
    if scale_data:
        data_std = data.apply(scale)
    else:
        data_std = data
    corrs = np.asarray(data_std.cov())
    pca = PCA(n_components = 1).fit(corrs)
    mkt_index = -scale(pca.transform(data_std))
    return mkt_index

price_index = make_pca_index(prices)
plt.figure(figsize=(17,5))
plt.subplot(121)
plt.plot(prices.index, scale(dji), 'k.')
plt.xlabel('PCA index')
plt.ylabel('Dow Jones Index')
ols_fit = OLSreg(scale(dji), price_index)
plt.plot(price_index, ols_fit.fittedvalues, 'r-',
         label = 'R2 = %4.3f' % round(ols_fit.rsquared, 3))
plt.legend(loc = 'upper left')
plt.subplot(122)
plt.plot(dates, price_index, label = 'PCA Price Index')
plt.plot(dates, scale(dji), label = 'DJ Index')
plt.legend(loc = 'upper left')
plt.savefig('price_index.png')