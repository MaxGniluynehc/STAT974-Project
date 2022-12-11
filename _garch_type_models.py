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
data_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/data/"
s = datetime(2000,1,1)
e = datetime.today()
df_name = "BTC"
df = pd.read_csv(data_PATH+"{}_from={}_to={}.csv".format(df_name, datetime(2000,1,1).date(), datetime.today().date()))
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
garch11 = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, q=1, dist="gaussian")
garch11_fitted = garch11.fit()


# ============================= EGARCH(1,1,1) =================================== #
egarch11 = arch_model(logr_train, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="gaussian")
egarch11_fitted = egarch11.fit()


egarch11_studentst= arch_model(logr_train, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="studentst")
egarch11_studentst_fitted = egarch11_studentst.fit()

egarch11_skewstudent= arch_model(logr_train, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="skewstudent")
egarch11_skewstudent_fitted = egarch11_skewstudent.fit()


# ============================= GJR-GARCH(1,1) =================================== #
gjrgarch11 = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="gaussian")
gjrgarch11_fitted = gjrgarch11.fit()

gjrgarch11_skewstudent = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewstudent")
gjrgarch11_skewstudent_fitted = gjrgarch11_skewstudent.fit()


# ============================= TGARCH(1,1) =================================== #
tgarch11 = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="gaussian", power=1)
tgarch11_fitted = tgarch11.fit()

tgarch11_skewstudent = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewstudent", power=1)
tgarch11_skewstudent_fitted = tgarch11_skewstudent.fit()


# ============================= FIGARCH(1,1) =================================== #
# egarch11 = arch_model(logr_train, mean="Constant", vol="FIGARCH", p=1, q=1, dist="gaussian")


# ============================= APARCH(1,1,1) =================================== #
aparch11 = arch_model(logr_train, mean="Constant", dist="skewstudent")
aparch_vol_process = arch.univariate.APARCH(p=1, o=1, q=1, delta=2)
aparch11.volatility = aparch_vol_process
aparch11_fitted = aparch11.fit()


# ============================= EWMA =================================== #
ewma = arch_model(logr_train, dist="gaussian")
ewma_vol_process = arch.univariate.EWMAVariance()
ewma.volatility = ewma_vol_process
ewma_fitted = ewma.fit()

ewma_skewstudent = arch_model(logr_train, dist="skewstudent")
ewma_skewstudent.volatility = ewma_vol_process
ewma_skewstudent_fitted = ewma_skewstudent.fit()





