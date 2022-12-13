import numpy as np
import torch as tc
import pandas as pd
import arch
import yfinance as yf
# from yahoofinancials import YahooFinancials
# import yahoo_finance as yf
from datetime import datetime, timedelta


data_PATH="/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/data/"

s = datetime(2000,1,1)
e = datetime.today()

SPX = yf.Ticker("^GSPC").history(period="1d", start=s, end=e)
Oil = yf.Ticker("CL=F").history(period="1d", start=s, end=e)
Gold = yf.Ticker("GC=F").history(period="1d", start=s, end=e)
BTC = yf.Ticker("BTC-USD").history(period="1d", start=s, end=e)
ETH = yf.Ticker("ETH-USD").history(period="1d", start=s, end=e)
CMC = yf.Ticker("^CMC200").history(period="1d", start=s, end=e)
Binance = yf.Ticker("BUSD-USD").history(period="1d", start=s, end=e)
NSDQ = yf.Ticker("NQ=F").history(period="1d", start=s, end=e)

SPX.to_csv(data_PATH + "SPX_from={}_to={}.csv".format(s.date(),e.date()))
Oil.to_csv(data_PATH + "Oil_from={}_to={}.csv".format(s.date(),e.date()))
Gold.to_csv(data_PATH + "Gold_from={}_to={}.csv".format(s.date(),e.date()))
BTC.to_csv(data_PATH + "BTC_from={}_to={}.csv".format(s.date(),e.date()))
ETH.to_csv(data_PATH + "ETH_from={}_to={}.csv".format(s.date(),e.date()))
CMC.to_csv(data_PATH + "CMC_from={}_to={}.csv".format(s.date(),e.date()))
Binance.to_csv(data_PATH + "Binance_from={}_to={}.csv".format(s.date(),e.date()))
NSDQ.to_csv(data_PATH + "NSDQ_from={}_to={}.csv".format(s.date(),e.date()))






