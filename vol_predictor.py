import numpy as np
import statsmodels.graphics.tsaplots
import torch as tc
from torch.nn import Linear,LSTM, LSTMCell, RNN, ReLU, Sigmoid, Tanh, Sequential, Flatten, Dropout1d
from torch.nn import MSELoss, L1Loss

# input data: [batch_size, sequence_length, input_size] = [bs, L, Hin]
# where sequence_length=21, input_size = param_size_of_GARCH + explain_variable_size


class VolPredictor(tc.nn.Module):
    def __init__(self, input_size, hidden_size=2, num_layers=3, rnn_type="lstm", device="cpu"):
        super(VolPredictor, self).__init__()
        self.Hin = input_size
        self.Hout = hidden_size
        self.nl = num_layers
        self.rnn_type = rnn_type
        self.device = device

        if self.rnn_type == "lstm" or self.rnn_type == "gew-lstm":
            # self.rnn = LSTM(input_size=self.Hin, hidden_size=self.Hout, num_layers=self.nl, batch_first=True).to(self.device)
            self.lstm1 = LSTMCell(input_size=self.Hin, hidden_size=10, device=self.device)
            self.lstm2 = LSTMCell(input_size=10, hidden_size=4, device=self.device)
            self.lstm3 = LSTMCell(input_size=4, hidden_size=self.Hout, device=self.device)
            # self.lstm1 = LSTMCell(input_size=self.Hin, hidden_size=64, device=self.device)
            # self.lstm2 = LSTMCell(input_size=64, hidden_size=32, device=self.device)
            # self.lstm3 = LSTMCell(input_size=32, hidden_size=self.Hout, device=self.device)


        elif self.rnn_type == "lstm_whole":
            self.lstm = LSTM(input_size=self.Hin, hidden_size=self.Hout, num_layers=self.nl, batch_first=True, device=self.device)

        elif self.rnn_type == "rnn":
            self.rnn = RNN(input_size=self.Hin, hidden_size=self.Hout, num_layers=self.nl, batch_first=True).to(self.device)
        else:
            ValueError("Wrong rnn_type!")
        # Input: [bs, L, Hin]
        # Output: [bs, L, Hout]

        self.fc1 = Linear(self.Hout, 5).to(self.device)
        self.fc2 = Linear(5, 1).to(self.device)

        self.relu = ReLU().to(self.device)
        self.sigmoid = Sigmoid().to(self.device)
        self.tanh = Tanh().to(self.device)
        self.flatten = Flatten().to(self.device)

        self.drop1 = Dropout1d(p=0.3).to(self.device)
        self.drop2 = Dropout1d(p=0.8).to(self.device)
        self.drop3 = Dropout1d(p=0.8).to(self.device)

        # self.MSE = MSELoss()
        # self.MAE = MAELoss()

    def forward(self, x, hidden):
        if self.rnn_type == "lstm" or self.rnn_type == "gew-lstm":
            # o = tc.empty([x.shape[0], x.shape[1], self.Hout], device=self.device) # [bs, L, Hout]
            (h1, c1), (h2, c2), (h3, c3) = hidden
            for i in range(x.shape[1]):
                h1, c1 = self.lstm1(x[:,i,:], (h1, c1)) # x[:,i,:]: [bs, Hin], h1: [bs, 10]
                h1 = self.drop1(h1)
                h2, c2 = self.lstm2(h1, (h2, c2)) # h2: [bs, 4]
                h2 = self.drop2(h2)
                h3, c3 = self.lstm3(h2, (h3, c3)) # h3: [bs, Hout]
                h3 = self.drop2(h3)
                # o[:,i,:] = h3
            out = Sequential(self.fc1, self.sigmoid,
                             self.fc2  # , self.relu
                             )(h3) # h3: [bs, 1]

        elif self.rnn_type == "lstm_whole":
            print("x shape={}, h shape={}, c_shape={}".format(x.shape, hidden[0].shape, hidden[0].shape))
            x, _ = self.lstm(x, hidden) # x: [bs, L, Hout]
            x_ = x.view([x.shape[0], -1, self.Hout]).sum(dim=1)
            out = Sequential(self.fc1, self.sigmoid,
                             self.fc2, self.relu,
                             self.flatten)(x_)  # x: [bs, 1]

        elif self.rnn_type == "rnn":
            x, _ = self.rnn(x, hidden) # x: [bs, L, Hout]
            out = Sequential(self.fc1, self.sigmoid,
                           self.fc2, self.relu,
                           self.flatten)(x) # x: [bs, L]
        return out

    def init_hidden(self, batch_size):
        if self.rnn_type == "lstm":
            h1 = tc.ones(size=[batch_size, 10], device=self.device) * 1e-4
            c1 = tc.ones(size=[batch_size, 10], device=self.device) * 1e-4
            h2 = tc.ones(size=[batch_size, 4], device=self.device) * 1e-4
            c2 = tc.ones(size=[batch_size, 4], device=self.device) * 1e-4

            # h1 = tc.ones(size=[batch_size, 64], device=self.device) * 1e-4
            # c1 = tc.ones(size=[batch_size, 64], device=self.device) * 1e-4
            # h2 = tc.ones(size=[batch_size, 32], device=self.device) * 1e-4
            # c2 = tc.ones(size=[batch_size, 32], device=self.device) * 1e-4
            h3 = tc.ones(size=[batch_size, self.Hout], device=self.device) * 1e-4
            c3 = tc.ones(size=[batch_size, self.Hout], device=self.device) * 1e-4
            return [(h1, c1), (h2, c2), (h3, c3)]

        elif self.rnn_type == "gew-lstm":
            h1 = tc.zeros(size=[batch_size, 10], device=self.device) # * 1e-4
            c1 = tc.zeros(size=[batch_size, 10], device=self.device) # * 1e-4
            h2 = tc.zeros(size=[batch_size, 4], device=self.device) # * 1e-4
            c2 = tc.zeros(size=[batch_size, 4], device=self.device) # * 1e-4
            h3 = tc.zeros(size=[batch_size, self.Hout], device=self.device) # * 1e-4
            c3 = tc.zeros(size=[batch_size, self.Hout], device=self.device) # * 1e-4
            return [(h1, c1), (h2, c2), (h3, c3)]


        elif self.rnn_type == "lstm_whole":
            h = tc.ones(size=[self.nl, batch_size, self.Hout], device=self.device) * 1e-4
            c = tc.ones(size=[self.nl, batch_size, self.Hout], device=self.device) * 1e-4
            return (h,c)

        elif self.rnn_type == "rnn":
            h = tc.ones(size=[self.nl, batch_size, self.Hout], device=self.device) * 1e-4
            return h


    # def loss(self):


if __name__ == '__main__':
    bs, L, Hin, Hout, nlayer, rnn_type = (128, 21, 15, 2, 3, "lstm")
    dev = tc.device("mps")
    m = VolPredictor(Hin, Hout, nlayer, rnn_type, dev)
    hidden = m.init_hidden(bs)
    d = tc.randn([bs, L, Hin], device=m.device)
    d[:, 1,:].shape

    o = tc.empty([d.shape[0], d.shape[1], 2])


    h = tc.randn([d.shape[0], 2])
    o[:,2,:] = h

    x, _ = m.rnn(d, hidden)
    out = m.forward(x=d, hidden=hidden)
    # out.shape
















