import torch as tc
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
from arch import arch_model

# from _garch_type_models import df, df_train, df_test, logr, logr_train, logr_test, realized_vol,\
# from _garch_type_models import gjrgarch11_skewstudent_fitted

from vol_predictor import VolPredictor
from dataloader import BTCDataset

from train_GJR_LSTM import realized_vol, logr_df

from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_lm



tc.random.manual_seed(210204)

Hin, Hout, rnn_type, device, garch_type, epochs, lr, batch_size, bws = \
    (logr_df.shape[-1] + 4, 2, "lstm", "mps", "GJR", 50, 1e-4, 128, 21)

if garch_type is None:
    Hin = logr_df.shape[-1]  # 6
elif garch_type == "GJR":
    Hin = logr_df.shape[-1] + 4  # 10
elif garch_type == "GJR-EXP-EWMA":
    Hin = logr_df.shape[-1] + 4 + 4 + 1  # 15

# logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221212/"
# logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221213_ld/"
# logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221213/" # update at each batch
# logs_PATH = "/Users/y222chen/Documents/Max/Study/STAT974_Econometrics/Project/project/logs20221214_{}/".format(garch_type)
# logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221214_GJR/"

# logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221215_GJR/"
logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221215_T/"

figs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/figs/result-lstm/"

# volpredictor = VolPredictor(input_size=Hin, hidden_size=Hout, num_layers=3, rnn_type=rnn_type, device=device)
# volpredictor.load_state_dict(tc.load(logs_PATH + "trained_volpredictor_at_epoch={}.pth".format(79)).state_dict())
# volpredictor = tc.load(logs_PATH+"trained_volpredictor_at_epoch={}.pth".format(9))

# volpredictor = tc.load(logs_PATH + "trained_volpredictor_at_epoch={}.pth".format(79)) # in 20221214_GJR
# volpredictor = tc.load(logs_PATH + "trained_volpredictor_at_epoch={}.pth".format(62)) # in 20221215_GJR
volpredictor = tc.load(logs_PATH + "trained_volpredictor_at_epoch={}.pth".format(43)) # in 20221215_T


ds = tc.tensor(logr_df.values, dtype=tc.float32, device=device)
ds_train = ds[:int(np.argwhere(logr_df.index == datetime(2021, 1, 1))), :]
ds_test = ds[int(np.argwhere(logr_df.index == datetime(2021, 1, 1))):, :]
# dataset = BTCDataset(ds_train, garch_type=garch_type)
# dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
RV = tc.tensor(realized_vol).to(dtype=tc.float32, device=device)


ds_in_train = tc.load(logs_PATH+"ds_in.pt")
hidden_init_train = volpredictor.init_hidden(ds_in_train.shape[0])
vol_pred_train = volpredictor.forward(ds_in_train, hidden_init_train)
tc.save(vol_pred_train, logs_PATH+"vol_pred_train.pt")
resid = vol_pred_train.detach().cpu().numpy().flatten() - realized_vol[-ds_in_train.shape[0]:].values



dataset_test = BTCDataset(ds_test, garch_type=garch_type)
dataloader_test = DataLoader(dataset_test, batch_size=dataset_test.__len__(), drop_last=True)
# _, ds_in_test = next(enumerate(dataloader_test))
# tc.save(ds_in_test, logs_PATH+"ds_in_test.pt")
ds_in_test = tc.load(logs_PATH +"ds_in_test.pt" )
ds_in_test.shape


hidden_init_test = volpredictor.init_hidden(ds_in_test.shape[0])
vol_pred_test = volpredictor.forward(ds_in_test, hidden_init_test)
tc.save(vol_pred_test, logs_PATH+"vol_pred_test.pt")
# vol_pred_test = tc.load(logs_PATH+"vol_pred_test.pt")



plt.figure()
plt.plot(logr_df.index[-ds_in_test.shape[0]:], vol_pred_test.detach().cpu().numpy(), label="pred_vol")
plt.plot(realized_vol[-ds_in_test.shape[0]:], label="realized_vol")
ticks = [round((vol_pred_test.shape[0]-1)*q) for q in [0, 0.25, 0.5, 0.75, 1]]
plt.xticks(logr_df.index[-ds_in_test.shape[0]:][ticks],
           logr_df.index[-ds_in_test.shape[0]:][ticks].date)
plt.legend()



lstm_valid = pd.DataFrame(index=["jarque-bera",
                                  "Ljung-box-res(5)", "Ljung-box-res(10)",
                                  "Ljung-box-res(15)", "Ljung-box-res(20)",
                                  "Box-pierce-res(5)", "Boxpierce-res(10)",
                                  "Box-pierce-res(15)", "Box-pierce-res(20)",
                                  "Ljung-box-res^2(5)", "Ljung-box-res^2(10)",
                                  "Ljung-box-res^2(15)", "Ljung-box-res^2(20)",
                                  "Box-pierce-res^2(5)", "Boxpierce-res^2(10)",
                                  "Box-pierce-res^2(15)", "Box-pierce-res^2(20)",
                                  "ArchEffect-res",
                                  "MSE", "HMSE",
                                  "MAE", "HMAE", "Correlation"],
                           columns=["GJR-LSTM"])



lstm_valid1 = pd.DataFrame(index=["jarque-bera",
                                  "Ljung-box-res(5)", "Ljung-box-res(10)",
                                  "Ljung-box-res(15)", "Ljung-box-res(20)",
                                  "Box-pierce-res(5)", "Boxpierce-res(10)",
                                  "Box-pierce-res(15)", "Box-pierce-res(20)",
                                  "Ljung-box-res^2(5)", "Ljung-box-res^2(10)",
                                  "Ljung-box-res^2(15)", "Ljung-box-res^2(20)",
                                  "Box-pierce-res^2(5)", "Boxpierce-res^2(10)",
                                  "Box-pierce-res^2(15)", "Box-pierce-res^2(20)",
                                  "ArchEffect-res",
                                  "MSE", "HMSE",
                                  "MAE", "HMAE", "Correlation"],
                           columns=["TGARCH-LSTM"])

for idx, m in enumerate([0]):
    col_idx = []
    jb = jarque_bera(resid)
    col_idx.append("{}({})".format(round(jb[0],3), round(jb[1],3)))

    std_resid = (resid - resid.mean())/resid.std()

    lb1 = acorr_ljungbox(std_resid, lags=[5, 10, 15, 20], boxpierce=True)
    for i in range(4):
        col_idx.append("{}({})".format(round(lb1.iloc[i, 0],3), round(lb1.iloc[i, 1],3)))
    for i in range(4):
        col_idx.append("{}({})".format(round(lb1.iloc[i, 2],3), round(lb1.iloc[i, 3],3)))

    lb2 = acorr_ljungbox(std_resid ** 2, lags=[5, 10, 15, 20], boxpierce=True)
    for i in range(4):
        col_idx.append("{}({})".format(round(lb2.iloc[i, 0], 3), round(lb2.iloc[i, 1], 3)))
    for i in range(4):
        col_idx.append("{}({})".format(round(lb2.iloc[i, 2], 3), round(lb2.iloc[i, 3], 3)))

    lm = acorr_lm(std_resid)
    col_idx.append("{}({})".format(round(lm[0], 3), round(lm[1], 3)))

    vol_pred = vol_pred_test.flatten().detach().cpu().numpy()
    # vol_pred.shape
    realized_vol_test = realized_vol[-ds_in_test.shape[0]:].values
    ones = np.ones(shape=realized_vol_test.shape)
    mse = np.mean(np.square(vol_pred - realized_vol_test))
    hmse = np.mean(np.square(ones - vol_pred/realized_vol_test))
    mae = np.mean(np.abs(vol_pred - realized_vol_test))
    hmae = np.mean(np.abs(ones - vol_pred/realized_vol_test))
    col_idx.append(str(round(mse, 6)))
    col_idx.append(str(round(hmse, 6)))
    col_idx.append(str(round(mae, 6)))
    col_idx.append(str(round(hmae, 6)))

    col_idx.append(round(np.corrcoef(vol_pred, realized_vol_test)[0,1], 4))

    # col_idx.append("({}, {})".format(round(m.aic, 2), round(m.bic, 2)))

    lstm_valid1.iloc[:, idx] = col_idx
lstm_valid1.to_csv(figs_PATH + "{}-model-validation.csv".format("tgarch-lstm"))

lstm_valid["TGARCH-LSTM"] = lstm_valid1["TGARCH-LSTM"]


print(lstm_valid.to_latex())



# Out-of-sample-prediction
plt.figure()
plt.plot(realized_vol[-ds_in_test.shape[0]:], color="gray", alpha=0.5, label="realized_vol")
plt.plot(realized_vol[-ds_in_test.shape[0]:].index, vol_pred_test.flatten().detach().cpu().numpy(),
         label="pred_cond_vol", color="red")
ticks = [round((vol_pred_test.shape[0]-1)*q) for q in [0, 0.25, 0.5, 0.75, 1]]
plt.xticks(logr_df.index[-ds_in_test.shape[0]:][ticks],
           logr_df.index[-ds_in_test.shape[0]:][ticks].date)
plt.legend()
# plt.title("Conditional Variance (test)")
plt.savefig(figs_PATH + "{}_out_of_sample_vol_pred.png".format("TGARCH-LSTM"))

# np.sum(np.square(vol_pred_test.detach().cpu().numpy().flatten() - realized_vol[-ds_in_test.shape[0]:].values))



# Convergence of loss functions

ll = np.loadtxt(logs_PATH+"logs_to_epoch=49.txt")
plt.figure()
plt.plot(np.arange(ll.shape[0])/round(ds_in_train.shape[0]/batch_size), ll, color="black")
plt.xlim([0,33])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig(figs_PATH+"loss_decay_{}.png".format("TGARCH_LSTM"))








