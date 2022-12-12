import numpy as np
import statsmodels.graphics.tsaplots
import torch as tc
from torch.nn import Linear,LSTM, RNN, ReLU, Sigmoid, Tanh, Sequential, Flatten


# input data: [batch_size, sequence_length, input_size] = [bs, L, Hin]
# where sequence_length=21, input_size = param_size_of_GARCH + explain_variable_size


class VolPredictor(tc.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type, device):
        super(VolPredictor, self).__init__()
        self.Hin = input_size
        self.Hout = hidden_size
        self.nl = num_layers
        self.rnn_type = rnn_type
        self.device = device

        if rnn_type == "lstm":
            self.rnn = LSTM(input_size=self.Hin, hidden_size=self.Hout, num_layers=self.nl, batch_first=True).to(self.device)
        elif rnn_type == "rnn":
            self.rnn = RNN(input_size=self.Hin, hidden_size=self.Hout, num_layers=self.nl, batch_first=True).to(self.device)
        else:
            ValueError("Wrong rnn_type!")
        # Input: [bs, L, Hin]
        # Output: [bs, L, Hout]

        self.fc1 = Linear(self.Hout, 128).to(self.device)
        self.fc2 = Linear(128, 1).to(self.device)

        self.relu = ReLU().to(self.device)
        self.sigmoid = Sigmoid().to(self.device)
        self.tanh = Tanh().to(self.device)
        self.flatten = Flatten().to(self.device)

    def forward(self, x, hidden:tuple[tc.Tensor, tc.Tensor] or tc.Tensor):
        x, _ = self.rnn(x, hidden) # x: [bs, L, Hout]
        x = Sequential(self.fc1, self.sigmoid,
                       self.fc2, self.relu,
                       self.flatten)(x) # x: [bs, L]
        return x

    def init_hidden(self, batch_size):
        h = tc.ones([self.nl, batch_size, self.Hout], device=self.device)*1e-4
        c = tc.ones([self.nl, batch_size, self.Hout], device=self.device)*1e-4
        if self.rnn_type == "lstm":
            return (h,c)
        elif self.rnn_type == "rnn":
            return h

    # def loss(self):


if __name__ == '__main__':
    bs, L, Hin, Hout, nlayer, rnn_type = (128, 21, 15, 64, 3, "lstm")
    dev = tc.device("mps")
    m = VolPredictor(Hin, Hout, nlayer, rnn_type, dev)
    hidden = m.init_hidden(bs)
    d = tc.randn([bs, L, Hin], device=m.device)
    x, _ = m.rnn(d, hidden)
    out = m.forward(x=d, hidden=hidden)
















