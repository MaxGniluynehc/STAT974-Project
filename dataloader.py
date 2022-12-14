import arch
from arch import arch_model
import torch as tc
from torch.utils.data import Dataset, DataLoader

class BTCDataset(Dataset):
    def __init__(self, price_scenario, batch_time_size=21, garch_type = None):
        '''
        :param price_scenario: [T, 6]
        :param batch_time_size: the length of time in each batched sample.
            (Since the data is time series, we cannot randomly sample over time. So the dataloader
            has to randomly select a single starting point (t between 0 to T), from which a batch
            of time-series chunk of length batch_time_size is sampled as a whole.)
        '''
        self.T = price_scenario.shape[0]
        self.price_scenario = price_scenario
        self.batch_time_size = batch_time_size
        self.garch_type = garch_type

    def __len__(self):
        return self.T - self.batch_time_size + 1

    def __getitem__(self, index):
        ps = self.price_scenario[index: self.batch_time_size+index, :] # [bws, 6] (bws = 21)
        if self.garch_type is None:
            batch = ps # [bws, 6]

        elif self.garch_type == "GJR":
            btc = ps[:, 0] # [21]
            garch = arch_model(btc.cpu().numpy(), mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewstudent", rescale=False)
            garch_params = tc.tensor(garch.fit(show_warning=False).params[["omega", "alpha[1]", "gamma[1]", "beta[1]"]])\
                .to(dtype=tc.float32, device=ps.device).repeat([self.batch_time_size, 1])
            # p = garch_params.repeat([self.batch_time_size, 1])
            batch = tc.concat((garch_params, ps), dim=1) # [bws, 10] (bws = 21)

        elif self.garch_type == "GJR-EXP-EWMA":
            btc = ps[:, 0] # [21]
            GJR = arch_model(btc.cpu().numpy(), mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewstudent", rescale=False)
            GJR_params = tc.tensor(GJR.fit(show_warning=False).params[["omega", "alpha[1]", "gamma[1]", "beta[1]"]])\
                .to(dtype=tc.float32, device=ps.device).repeat([self.batch_time_size, 1])

            EXP = arch_model(btc.cpu().numpy(), mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="skewstudent", rescale=False)
            EXP_params = tc.tensor(EXP.fit(show_warning=False).params[["omega", "alpha[1]", "gamma[1]", "beta[1]"]])\
                .to(dtype=tc.float32, device=ps.device).repeat([self.batch_time_size, 1])

            EWMA = arch_model(btc.cpu().numpy(), dist="skewstudent", rescale=False)
            EWMA.volatility = arch.univariate.EWMAVariance(lam=None)
            EWMA_params = tc.tensor(EWMA.fit(show_warning=False).params["lam"])\
                .to(dtype=tc.float32, device=ps.device).repeat([self.batch_time_size, 1])

            batch = tc.concat((GJR_params, EXP_params, EWMA_params, ps), dim=1) # [bws, 15] (bws = 21)

        else:
            ValueError("Wrong garch_type!")

        return batch
