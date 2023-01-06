
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

# data_PATH="/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/data/"
# data_PATH = "/Users/y222chen/Documents/Max/Study/STAT974_Econometrics/Project/project/data/"

data_PATH="~/STAT974-Project/data/" # working dictory on linux

s = datetime(2000,1,1)
e = datetime(2022, 12, 12) #today()
# garch_params = gjrgarch11_skewstudent_fitted.params


# ============================== Load Data =====================================#
BTC = pd.read_csv(data_PATH+"BTC_from={}_to={}.csv".format(s.date(),e.date()), index_col="Date")
SPX = pd.read_csv(data_PATH+"SPX_from={}_to={}.csv".format(s.date(),e.date()), index_col="Date")
NSDQ = pd.read_csv(data_PATH+"NSDQ_from={}_to={}.csv".format(s.date(),e.date()), index_col="Date")
Oil = pd.read_csv(data_PATH+"Oil_from={}_to={}.csv".format(s.date(),e.date()), index_col="Date")
Gold = pd.read_csv(data_PATH+"Gold_from={}_to={}.csv".format(s.date(),e.date()), index_col="Date")
TBill = pd.read_csv(data_PATH+"DTB3.csv", index_col="DATE").replace(".", np.nan).astype(np.float64)


df = pd.DataFrame(BTC.Close.values, index=BTC.index, columns=["BTC"])
df["SPX"] = SPX.Close
df["NSDQ"] = NSDQ.Close
df["Oil"] = Oil.Close
df["Gold"] = Gold.Close
df["TBill"] = TBill.DTB3
df.index = pd.DatetimeIndex(df.index)
df = df.interpolate(method="time")

logr_df = pd.DataFrame([], index=df.index[:-1], columns=df.keys())
logr_df["BTC"] = np.log(df.BTC.values[1:]/df.BTC.values[:-1])
logr_df["SPX"] = np.log(df.SPX.values[1:]/df.SPX.values[:-1])
logr_df["NSDQ"] = np.log(df.NSDQ.values[1:]/df.NSDQ.values[:-1])
logr_df["Oil"] = np.log(df.Oil.values[1:]/df.Oil.values[:-1])
logr_df["Gold"] = np.log(df.Gold.values[1:]/df.Gold.values[:-1])
logr_df["TBill"] = df.TBill/100
logr_df = logr_df.dropna(how="any")
# any(logr_df.isna())


realized_vol = logr_df.BTC.rolling(window=21).std(ddof=0)



# ============================== Define model, optimizer, train-test ds =====================================#
if __name__ == '__main__':
    tc.random.manual_seed(210203040333)

    Hin, Hout, rnn_type, device, garch_type, epochs, lr, batch_size, bws = \
   (logr_df.shape[-1]+4, 2, "lstm", "cpu", "T", 50, 1e-3, 128, 21)  # changed to cuda for linux server

    if garch_type is None:
        Hin = logr_df.shape[-1]         # 6
    elif garch_type == "GJR":
        Hin = logr_df.shape[-1]+4       # 10
    elif garch_type == "GJR-EXP-EWMA":
        Hin = logr_df.shape[-1]+4+4+1   # 15

    # logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221212/"
    # logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221213_ld/"
    # logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221213/" # update at each batch
    # logs_PATH = "/Users/y222chen/Documents/Max/Study/STAT974_Econometrics/Project/project/logs20221214_{}/".format(garch_type)
    # logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221215_GJR/"
    # logs_PATH = "/Users/maxchen/Documents/Study/STA/STAT974_Econometrics/Project/project/logs20221215_T/"
    logs_PATH = "/u/y222chen/STAT974-Project/logs20230105_T/"


    volpredictor = VolPredictor(input_size=Hin, hidden_size=Hout, num_layers=3, rnn_type=rnn_type, device=device)
    # volpredictor.load_state_dict(tc.load(logs_PATH+"trained_volpredictor_at_epoch={}.pth".format(79)).state_dict())
    # volpredictor = tc.load(logs_PATH+"trained_volpredictor_at_epoch={}.pth".format(9))

    opt = Adam(volpredictor.parameters(), lr=lr)
    ds = tc.tensor(logr_df.values, dtype=tc.float32, device=device)
    tc.save(ds, logs_PATH + "ds.pt")

    ds_train = ds[:int(np.argwhere(logr_df.index == datetime(2021,1,1))), :]
    ds_test = ds[int(np.argwhere(logr_df.index == datetime(2021,1,1))):, :]
    dataset = BTCDataset(ds_train, garch_type=garch_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    RV = tc.tensor(realized_vol).to(dtype=tc.float32, device=device)
    # idx, db = next(enumerate(dataloader))
    # db.shape

    # ============================== Train =====================================#

    # def get_ds_in(garch_type = garch_type):
    #     btc = ds_train[:, 0]
    #     garch = arch_model(btc.cpu().numpy(), mean="Constant", vol="GARCH",
    #                        p=1, o=1, q=1, dist="skewstudent", rescale=False)
    #     garch_params = tc.tensor(garch.fit(show_warning=False).params).to(dtype=tc.float32, device=ds_train.device)
    #     p = garch_params[1:5].repeat([ds_train.shape[0], 1])
    #     # ds_in = tc.concat((p, ds_train), dim=1).unsqueeze(1)  # [1, total len, 10]
    #
    #     ds_in = tc.empty([ds_train.shape[0]-bws, bws, p.shape[1]+ds_train.shape[1]]).to(dtype=tc.float32, device=ds_train.device)
    #     k = tc.concat((p, ds_train), dim=1).to(dtype=tc.float32, device=ds_train.device)
    #     for j in range(k.shape[0] - bws):
    #         ds_in[j,:,:] = k[j:j+bws,:]
    #     # ds_in.shape
    #     return ds_in
    #
    # ds_in = get_ds_in()

    dataloader_full = DataLoader(dataset, batch_size=dataset.__len__(), drop_last=True)
    _, ds_in = next(enumerate(dataloader_full))
    tc.save(ds_in, logs_PATH+"ds_in.pt")
    ds_in = tc.load(logs_PATH+"ds_in.pt")


    volpredictor.train()
    loss_MSE = MSELoss()
    loss_L1 = L1Loss()
    loss_CE = CrossEntropyLoss()
    logs = tc.empty(0)
    # logs = tc.tensor(np.loadtxt(logs_PATH+"logs_to_epoch={}.txt".format(99))).to(dtype=tc.float32, device="cpu")


    for epoch in tqdm(range(0, 0+epochs)):
        l = 0
        opt.zero_grad()
        h_init = volpredictor.init_hidden(batch_size=batch_size)
        vols = tc.empty(0).to(device=volpredictor.device)
        rvs = tc.empty(0).to(device=volpredictor.device)
        for idx, db in enumerate(dataloader):
            # hidden[0][0].shape
            vol = volpredictor.forward(db, hidden=h_init)
            vols = tc.cat((vols, vol), dim=1)
            # rvs = tc.cat((rvs, RV[idx+bws-1].repeat(vol.shape)), dim=1)
            # vols.shape
            # rvs.shape
            # vol.shape
            # RV[idx + bws - 1].repeat(vol.shape).shape

            rv_dm = RV[idx+bws-1].repeat(vol.shape) - tc.mean(RV[idx+bws-1].repeat(vol.shape))
            vol_dm = vol - vol.mean()
            corr = tc.multiply(vol_dm, rv_dm).sum()/(tc.sqrt(max(tc.sum(tc.pow(vol_dm,2)), tc.tensor([1e-6], device=volpredictor.device)))
                                                     * tc.sqrt(max(tc.sum(tc.pow(rv_dm,2)), tc.tensor([1e-6], device=volpredictor.device))))

            ones = tc.ones(size=vol.shape, dtype=tc.float32, device=volpredictor.device)
            l = loss_MSE(ones, vol/RV[idx+bws-1].repeat(vol.shape)) # \
                # + loss_L1(vol, RV[idx+bws-1].repeat(vol.shape)) \
                # + loss_MSE(vol, RV[idx+bws-1].repeat(vol.shape)) \
                # + loss_L1(ones, vol/RV[idx+bws-1].repeat(vol.shape)) \
                # - corr.item()
            # + loss_L1(vol, RV[idx+bws-1].repeat(vol.shape))
            # + loss_CE(vol, RV[idx+bws-1].repeat(vol.shape))
            # - corr.item()

            l.backward()
            opt.step()
            logs = tc.concat((logs, l.detach().unsqueeze(-1).cpu()))

        # ones = tc.ones(size=vols.shape, dtype=tc.float32, device=volpredictor.device)
        # l = loss(ones, vols/rvs) # + loss(vols, rvs) + loss_L1(ones, vols/rvs) + loss_L1(vols, rvs)
        # # l = loss(vols, rvs)
        # l.backward()
        # opt.step()
        # logs = tc.concat((logs, l.detach().unsqueeze(-1).cpu()))

        if (epoch+1)%1 == 0:
            # hidden = volpredictor.init_hidden(batch_size=ds_train.shape[0])
            # h0 = tc.ones([3, volpredictor.Hout])*1e-4
            # c0 = tc.ones([3, volpredictor.Hout]) * 1e-4

            hidden_init = volpredictor.init_hidden(ds_in.shape[0])

            # a,_ = volpredictor.lstm(ds_in, (h0, c0))
            # a.shape
            # a.view([ds_in.shape[0], -1, volpredictor.Hout])
            # vol_pred = tc.nn.Sequential(volpredictor.fc1, volpredictor.sigmoid,
            #                  volpredictor.fc2, volpredictor.relu,
            #                  volpredictor.flatten)(a.view([ds_in.shape[0], -1, volpredictor.Hout]))
            # vol_pred.shape
            vol_pred = volpredictor.forward(ds_in, hidden_init)
            # vol_pred.shape
            plt.figure()
            plt.plot(logr_df.index[bws-1:ds_in.shape[0]+bws-1], vol_pred.detach().cpu().numpy(), label= "pred_vol")
            plt.plot(realized_vol[bws-1:ds_in.shape[0]+bws-1], label="realized_vol")
            plt.legend()
            plt.savefig(logs_PATH+"pred_vs_realized_vol_epoch={}.png".format(int(epoch)))

            tc.save(volpredictor, logs_PATH+"trained_volpredictor_at_epoch={}.pth".format(epoch))

        tqdm(epochs).set_description('MSE Loss: %.8f' % (l.item()))

    np.savetxt(logs_PATH+"logs_to_epoch={}.txt".format(int(epoch)), logs.detach().numpy())


    # losses = np.loadtxt(logs_PATH+"logs_to_epoch={}.txt".format(int(49)))
    # plt.figure()
    # plt.plot(losses)
    # plt.ylim([-0.1, 1])



























