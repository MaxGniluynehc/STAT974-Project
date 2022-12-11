import numpy as np
import statsmodels.graphics.tsaplots
import torch as tc
import pandas as pd
import arch
from arch import arch_model

from _garch_type_models import *

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


# ============================= Exploratory Data Analysis =================================== #
# price process
plt.figure()
plt.plot(df.Close)
# plt.xticks(ticks=[0, int(df.shape[0]/2), df.shape[0]-1], labels=df.index[[0, int(df.shape[0]/2), df.shape[0]-1]])
plt.title("{} price process".format(df_name))

# (log) return process
logr = np.log(df.Close.iloc[1:].values/df.Close.iloc[:-1].values)
logr = pd.Series(logr, index=df.index[1:])
plt.figure()
plt.plot(logr)
# plt.xticks(ticks=[0, int(logr.shape[0]/2), logr.shape[0]-1], labels=logr.index[[0, int(logr.shape[0]/2), logr.shape[0]-1]])
plt.title("{} (log) return process".format(df_name))

# ACFs of returns and squared returns
plot_acf(logr, auto_ylims=True)
plt.title("ACF of returns")

plot_acf(logr**2, auto_ylims=True)
plt.title("ACF of squared returns")


# ============================= Train-Test Split =================================== #
df_train = df.loc[:datetime(2020,12,31), :]
df_test = df.loc[datetime(2021,1,1):, :]
logr_train = logr.loc[:datetime(2020,12,31)]
logr_test = logr.loc[datetime(2021,1,1):]
realized_vol_train = realized_vol.loc[:datetime(2020,12,31)]
realized_vol_test = realized_vol.loc[datetime(2021,1,1):]


# ============================== GARCH(1,1) =================================== #
# garch11 = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, q=1, dist="gaussian")
# fitted_garch11 = garch11.fit()

garch11_fitted.plot() # stdres + cond vol

qqplot(garch11_fitted.resid, line="s")
plt.title("Standardized Residuals")

jarque_bera(garch11_fitted.resid)

acorr_ljungbox(garch11_fitted.std_resid, lags=[5, 10, 15, 20], boxpierce=True)

acorr_ljungbox(garch11_fitted.std_resid**2, lags=[5, 10, 15, 20], boxpierce=True)

acorr_lm(garch11_fitted.std_resid)


plt.figure()
# plt.plot(logr_train**2, label="sqrd_return", color="gray", alpha=0.5)
plt.plot(realized_vol_train, label="realized_vol", color="gray")
plt.plot(garch11_fitted.conditional_volatility, label="cond_vol", color="red")
# plt.ylim([-0.01, 0.08])
plt.legend()
plt.title("Conditional Variance (train)")


plt.figure()
plt.plot(logr_test**2, color="gray", alpha=0.5, label = "sqrd_return")
pred = garch11_fitted.forecast(start= garch11_fitted.conditional_volatility.index[-1],
                               horizon= len(logr_test), #len(garch11_fitted.conditional_volatility),
                               reindex=False)
plt.plot(logr_test.index, pred.variance.values[0], label="pred_cond_vol", color="red")
plt.legend()
plt.title("Conditional Variance (test)")


pred_cond_vol = pd.Series(pred.variance.values[0], index=logr_test.index)
cond_vol_train_test = pd.concat([garch11_fitted.conditional_volatility**2, pred_cond_vol])
plt.figure()
# plt.plot(logr**2, color="gray", alpha=0.5, label="sqrd_return")
plt.plot(realized_vol, color="gray", label="realized_vol")
plt.plot(np.sqrt(cond_vol_train_test), label="cond_var", color="orange")
uncond_var = garch11_fitted.params[0]/(1-garch11_fitted.params[1]-garch11_fitted.params[2])
plt.hlines(y=np.sqrt(uncond_var), xmin=cond_vol_train_test.index[0], xmax=cond_vol_train_test.index[-1], colors="red", label="uncond_var")
# plt.ylim([-0.001, 0.05])
plt.legend()


# ============================= EGARCH(1,1,1) =================================== #
egarch11 = arch_model(logr_train, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="gaussian")
fitted_egarch11 = egarch11.fit()
fitted_egarch11.summary()


egarch11_studentst= arch_model(logr_train, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="studentst")
fitted_egarch11_studentst = egarch11_studentst.fit()
fitted_egarch11_studentst.summary()

egarch11_skewstudent= arch_model(logr_train, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="skewstudent")
fitted_egarch11_skewstudent = egarch11_skewstudent.fit()
fitted_egarch11_skewstudent.summary()



# ============================= GJR-GARCH(1,1) =================================== #
gjrgarch11 = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="gaussian")
gjrgarch11_fitted = gjrgarch11.fit()
gjrgarch11_fitted.summary()

gjrgarch11_skewstudent = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewstudent")
gjrgarch11_skewstudent_fitted = gjrgarch11_skewstudent.fit()
gjrgarch11_skewstudent_fitted.summary()


# ============================= TGARCH(1,1) =================================== #
tgarch11 = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="gaussian", power=1)
tgarch11_fitted = tgarch11.fit()
tgarch11_fitted.summary()

tgarch11_skewstudent = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewstudent", power=1)
tgarch11_skewstudent_fitted = tgarch11_skewstudent.fit()
tgarch11_skewstudent_fitted.summary()



# ============================= FIGARCH(1,1) =================================== #
egarch11 = arch_model(logr_train, mean="Constant", vol="FIGARCH", p=1, q=1, dist="gaussian")



# ============================= APARCH(1,1,1) =================================== #
aparch11 = arch_model(logr_train, mean="Constant", dist="skewstudent")
aparch_vol_process = arch.univariate.APARCH(p=1, o=1, q=1, delta=2)
aparch11.volatility = aparch_vol_process
aparch11.fit()


# ============================= EWMA =================================== #
ewma = arch_model(logr_train, dist="gaussian")
ewma_vol_process = arch.univariate.EWMAVariance()
ewma.volatility = ewma_vol_process
ewma_fitted = ewma.fit()
ewma_fitted.summary()

ewma_skewstudent = arch_model(logr_train, dist="skewstudent")
ewma_skewstudent.volatility = ewma_vol_process
ewma_skewstudent_fitted = ewma_skewstudent.fit()
ewma_skewstudent_fitted.summary()



# ============================= Model Comparison =================================== #
fitted_garch11.aic
fitted_garch11.bic
fitted_garch11.rsquared_adj



