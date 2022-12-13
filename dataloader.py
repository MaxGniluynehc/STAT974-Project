
from arch import arch_model
import torch as tc
from torch.utils.data import Dataset, DataLoader

class BTCDataset(Dataset):
    def __init__(self, price_scenario, batch_time_size=21):
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

    def __len__(self):
        return self.T - self.batch_time_size + 1

    def __getitem__(self, index):
        ps = self.price_scenario[index: self.batch_time_size+index, :] # [21, 6]
        btc = ps[:, 0] # [21]
        garch = arch_model(btc.cpu().numpy(), mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewstudent", rescale=False)
        garch_params = tc.tensor(garch.fit(show_warning=False).params).to(dtype=tc.float32, device=ps.device)
        p = garch_params[1:5].repeat([self.batch_time_size, 1])
        batch = tc.concat((p, ps), dim=1)
        return batch
