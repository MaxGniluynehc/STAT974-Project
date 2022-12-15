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
# data_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/data/"
# data_PATH = "/Users/y222chen/Documents/Max/Study/STAT974_Econometrics/Project/project/data/"
# s = datetime(2000,1,1)
# e = datetime(2022, 12, 12) #today()
# df_name = "BTC"
# df = pd.read_csv(data_PATH+"{}_from={}_to={}.csv".format(df_name, datetime(2000,1,1).date(), e.date()))
# df = df.set_index("Date")
# df.index = pd.DatetimeIndex(df.index)
# df.keys()


# ============================= Exploratory Data Analysis =================================== #
EDA_PATH = "/Users/y222chen/Documents/Max/Study/STAT974_Econometrics/Project/project/figs/EDA/"

# price process
plt.figure()
plt.plot(df.Close)
# plt.xticks(ticks=[0, int(df.shape[0]/2), df.shape[0]-1], labels=df.index[[0, int(df.shape[0]/2), df.shape[0]-1]])
# plt.title("{} price process".format(df_name))
plt.savefig(EDA_PATH + "BTC_price_process.png")

# (log) return process
logr = np.log(df.Close.iloc[1:].values/df.Close.iloc[:-1].values)
logr = pd.Series(logr, index=df.index[1:])
plt.figure()
plt.plot(logr)
# plt.xticks(ticks=[0, int(logr.shape[0]/2), logr.shape[0]-1], labels=logr.index[[0, int(logr.shape[0]/2), logr.shape[0]-1]])
# plt.title("{} (log) return process".format(df_name))
plt.savefig(EDA_PATH + "BTC_return_process.png")


BTC = pd.read_csv(data_PATH+"BTC_from={}_to={}.csv".format(s.date(),e.date()), index_col="Date")
SPX = pd.read_csv(data_PATH+"SPX_from={}_to={}.csv".format(s.date(),e.date()), index_col="Date")
NSDQ = pd.read_csv(data_PATH+"NSDQ_from={}_to={}.csv".format(s.date(),e.date()), index_col="Date")
Oil = pd.read_csv(data_PATH+"Oil_from={}_to={}.csv".format(s.date(),e.date()), index_col="Date")
Gold = pd.read_csv(data_PATH+"Gold_from={}_to={}.csv".format(s.date(),e.date()), index_col="Date")
TBill = pd.read_csv(data_PATH+"DTB3.csv", index_col="DATE").replace(".", np.nan).astype(np.float64)

dff = pd.DataFrame(BTC.Close.values, index=BTC.index, columns=["BTC"])
dff["SPX"] = SPX.Close
dff["NSDQ"] = NSDQ.Close
dff["Oil"] = Oil.Close
dff["Gold"] = Gold.Close
dff["TBill"] = TBill.DTB3
dff.index = pd.DatetimeIndex(dff.index)
dff = dff.interpolate(method="time")

logr_dff = pd.DataFrame([], index=dff.index[:-1], columns=dff.keys())
logr_dff["BTC"] = np.log(dff.BTC.values[1:]/dff.BTC.values[:-1])
logr_dff["SPX"] = np.log(dff.SPX.values[1:]/dff.SPX.values[:-1])
logr_dff["NSDQ"] = np.log(dff.NSDQ.values[1:]/dff.NSDQ.values[:-1])
logr_dff["Oil"] = np.log(dff.Oil.values[1:]/dff.Oil.values[:-1])
logr_dff["Gold"] = np.log(dff.Gold.values[1:]/dff.Gold.values[:-1])
logr_dff["TBill"] = dff.TBill/100
logr_dff = logr_dff.dropna(how="any")
# any(logr_dff.isna())


realized_vol = logr_dff.BTC.rolling(window=21).std(ddof=0)


fig, ax = plt.subplots(6,1, sharex=True, figsize = (5,7))
ax[0].plot(dff.BTC, label="BTC")
ax[0].legend()
ax[1].plot(dff.SPX, label="SPX")
ax[1].legend()
ax[2].plot(dff.NSDQ, label="NSDQ")
ax[2].legend()
ax[3].plot(dff.Oil, label="Oil")
ax[3].legend()
ax[4].plot(dff.Gold, label="Gold")
ax[4].legend()
ax[5].plot(dff.TBill, label="TBill")
ax[5].legend()
plt.savefig(EDA_PATH + "price_process.png")


fig, ax = plt.subplots(6,1, sharex=True, figsize = (5,7))
ax[0].plot(logr_dff.BTC, label="BTC")
ax[0].legend()
ax[1].plot(logr_dff.SPX, label="SPX")
ax[1].legend()
ax[2].plot(logr_dff.NSDQ, label="NSDQ")
ax[2].legend()
ax[3].plot(logr_dff.Oil, label="Oil")
ax[3].legend()
ax[4].plot(logr_dff.Gold, label="Gold")
ax[4].legend()
ax[5].plot(logr_dff.TBill, label="TBill")
ax[5].legend()
plt.savefig(EDA_PATH + "return_process.png")


plt.figure()
plt.plot(realized_vol, label="BTC")
plt.savefig(EDA_PATH + "BTC_volatility_process.png")


# ACFs of returns and squared returns
plot_acf(logr, auto_ylims=True)
# plt.title("ACF of returns")
plt.savefig(EDA_PATH + "ACF_of_BTC_returns.png")


plot_acf(logr**2, auto_ylims=True)
# plt.title("ACF of squared returns")
plt.savefig(EDA_PATH + "ACF_of_squared_BTC_returns.png")


# ============================= Train-Test Split =================================== #
df_train = df.loc[:datetime(2020,12,31), :]
df_test = df.loc[datetime(2021,1,1):, :]
logr_train = logr.loc[:datetime(2020,12,31)]
logr_test = logr.loc[datetime(2021,1,1):]
realized_vol_train = realized_vol.loc[:datetime(2020,12,31)]
realized_vol_test = realized_vol.loc[datetime(2021,1,1):]


# ============================== GARCH(1,1) =================================== #
# garch11 = arch_model(logr_train, mean="Constant", vol="GARCH", p=1, q=1, dist="gaussian")
# # garch11_fitted = garch11.fit()

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


# ============================= Model Comparison =================================== #
garch_fitted_list = [garch11_fitted, egarch11_fitted,
                     tgarch11_fitted, gjrgarch11_fitted,
                     aparch11_fitted, ewma_fitted]

garch_fitted_list_names = ["GARCH(1,1)", "EGARCH(1,1,1)",
                           "TGARCH(1,1,1)", "GJR-GARCH(1,1,1)",
                           "APARCH(1,1,1)", "EWMA"]

garch_skewstudent_fitted_list = [garch11_skewstudent_fitted, egarch11_skewstudent_fitted,
                                 tgarch11_skewstudent_fitted, gjrgarch11_skewstudent_fitted,
                                 aparch11_skewstudent_fitted, ewma_skewstudent_fitted]


m = next(iter(garch_skewstudent_fitted_list))

model_valid = pd.DataFrame(index=["jarque-bera",
                                  "Ljung-box-res(5)", "Ljung-box-res(10)",
                                  "Ljung-box-res(15)", "Ljung-box-res(20)",
                                  "Box-pierce-res(5)", "Boxpierce-res(10)",
                                  "Box-pierce-res(15)", "Box-pierce-res(20)",
                                  "Ljung-box-res^2(5)", "Ljung-box-res^2(10)",
                                  "Ljung-box-res^2(15)", "Ljung-box-res^2(20)",
                                  "Box-pierce-res^2(5)", "Boxpierce-res^2(10)",
                                  "Box-pierce-res^2(15)", "Box-pierce-res^2(20)",
                                  "ArchEffect",
                                  "MSE", "HMSE",
                                  "MAE", "HMAE"],
                           columns=garch_fitted_list_names)


for idx, m in enumerate(garch_skewstudent_fitted_list):
    col_idx = []
    jb = jarque_bera(m.resid)
    col_idx.append("{}({})".format(round(jb[0],3), round(jb[1],3)))

    lb1 = acorr_ljungbox(m.std_resid, lags=[5, 10, 15, 20], boxpierce=True)
    for i in range(4):
        col_idx.append("{}({})".format(round(lb1.iloc[i, 0],3), round(lb1.iloc[i, 1],3)))
    for i in range(4):
        col_idx.append("{}({})".format(round(lb1.iloc[i, 2],3), round(lb1.iloc[i, 3],3)))

    lb2 = acorr_ljungbox(m.std_resid ** 2, lags=[5, 10, 15, 20], boxpierce=True)
    for i in range(4):
        col_idx.append("{}({})".format(round(lb2.iloc[i, 0], 3), round(lb2.iloc[i, 1], 3)))
    for i in range(4):
        col_idx.append("{}({})".format(round(lb2.iloc[i, 2], 3), round(lb2.iloc[i, 3], 3)))

    lm = acorr_lm(m.std_resid)
    col_idx.append("{}({})".format(round(lm[0], 3), round(lm[1], 3)))

    pred = m.forecast(start=m.conditional_volatility.index[-1],
                      horizon=len(logr_test),  # len(garch11_fitted.conditional_volatility),
                      method="simulation",
                      reindex=False)
    vol_pred = np.sqrt(pred.variance.values).flatten()
    # vol_pred.shape
    ones = np.ones(shape=realized_vol_test.shape)
    mse = np.sum(np.square(vol_pred - realized_vol_test))
    hmse = np.sum(np.square(ones - vol_pred/realized_vol_test))
    mae = np.sum(np.abs(vol_pred - realized_vol_test))
    hmae = np.sum(np.abs(ones - vol_pred/realized_vol_test))
    col_idx.append(str(round(mse, 3)))
    col_idx.append(str(round(hmse, 3)))
    col_idx.append(str(round(mae, 3)))
    col_idx.append(str(round(hmae, 3)))

    model_valid.iloc[:, idx] = col_idx

print(model_valid.to_latex())



len(col_idx)



