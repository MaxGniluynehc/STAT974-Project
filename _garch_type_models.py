import numpy as np
import statsmodels.graphics.tsaplots
import torch as tc
import pandas as pd
import arch
from arch import arch_model
# import yfinance as yf
# from yahoofinancials import YahooFinancials
# import yahoo_finance as yf
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_predict
from statsmodels.graphics.gofplots import qqplot, qqline
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_lm


# ============================= Load Data =================================== #
# data_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/data/"
data_PATH = "/Users/y222chen/Documents/Max/Study/STAT974_Econometrics/Project/project/data/"
s = datetime(2000,1,1)
e = datetime(2022, 12, 12) # today()
# e = datetime.today()
df_name = "BTC"
df = pd.read_csv(data_PATH+"{}_from={}_to={}.csv".format(df_name, s.date(), e.date()))
df = df.set_index("Date")
df.index = pd.DatetimeIndex(df.index)
df.keys()


logr = np.log(df.Close.iloc[1:].values/df.Close.iloc[:-1].values)
logr = pd.Series(logr, index=df.index[1:])
realized_vol = logr.rolling(window=21).std(ddof=0)


# ============================= Train-Test Split =================================== #
df_train = df.loc[:datetime(2020,12,31), :]
df_test = df.loc[datetime(2021,1,1):, :]
logr_train = logr.loc[:datetime(2020,12,31)]
logr_test = logr.loc[datetime(2021,1,1):]



# ============================== GARCH(1,1) =================================== #
garch11 = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, q=1, dist="gaussian", rescale=False)
garch11_fitted = garch11.fit()

garch11_skewstudent = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, q=1, dist="skewstudent", rescale=False)
garch11_skewstudent_fitted = garch11_skewstudent.fit()


# ============================= EGARCH(1,1,1) =================================== #
egarch11 = arch_model(logr_train, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="gaussian", rescale=False)
egarch11_fitted = egarch11.fit()

egarch11_studentst= arch_model(logr_train, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="studentst", rescale=False)
egarch11_studentst_fitted = egarch11_studentst.fit()

egarch11_skewstudent= arch_model(logr_train, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="skewstudent", rescale=False)
egarch11_skewstudent_fitted = egarch11_skewstudent.fit()


# ============================= GJR-GARCH(1,1) =================================== #
gjrgarch11 = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="gaussian", rescale=False)
gjrgarch11_fitted = gjrgarch11.fit()
gjrgarch11_fitted.summary()
# gjrgarch11_fitted.params

gjrgarch11_skewstudent = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewstudent", rescale=False)
gjrgarch11_skewstudent_fitted = gjrgarch11_skewstudent.fit()
# gjrgarch11_skewstudent_fitted.params[1:5]

# ============================= TGARCH(1,1) =================================== #
tgarch11 = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="gaussian", power=1, rescale=False)
tgarch11_fitted = tgarch11.fit()

tgarch11_skewstudent = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewstudent", power=1, rescale=False)
tgarch11_skewstudent_fitted = tgarch11_skewstudent.fit()


# ============================= APARCH(1,1,1) =================================== #
aparch11 = arch_model(logr_train, mean="Constant", dist="gaussian", rescale=False)
aparch11.volatility = arch.univariate.APARCH(p=1, o=1, q=1, delta=2)
aparch11_fitted = aparch11.fit()


aparch11_skewstudent = arch_model(logr_train, mean="Constant", dist="skewstudent", rescale=False)
aparch11_skewstudent.volatility = arch.univariate.APARCH(p=1, o=1, q=1, delta=2)
aparch11_skewstudent_fitted = aparch11_skewstudent.fit()


# ============================= EWMA =================================== #
ewma = arch_model(logr_train, dist="gaussian", rescale=False)
ewma.volatility = arch.univariate.EWMAVariance(lam=None)
ewma_fitted = ewma.fit()
ewma_fitted.summary()

ewma_skewstudent = arch_model(logr_train, dist="skewstudent", rescale=False)
ewma_skewstudent.volatility = arch.univariate.EWMAVariance(lam=None)
ewma_skewstudent_fitted = ewma_skewstudent.fit()
ewma_skewstudent_fitted.summary()




