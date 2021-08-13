import torch
from torch import nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reparameter(mu, log_var):
    std = torch.exp(log_var * 0.5)
    esp = torch.randn_like(std)
    return mu + std * esp


# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.reformat = nn.Linear(in_features=hidden_size + 4, out_features=hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear_mean = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear_logvar = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # self.bn = nn.BatchNorm1d(num_features=1)

    def forward(self, input, hidden):
        hidden = self.reformat(hidden)
        embedded = self.embedding(input)

        for word_vec in embedded:
            tem = word_vec.view(1, 1, -1)
            output, hidden = self.gru(tem, hidden)

        # hidden = self.bn(hidden)
        mean = self.linear_mean(hidden)
        logvar = self.linear_logvar(hidden)
        return mean, logvar

    def initHidden(self, types):
        return torch.cat([torch.zeros(1, 1, self.hidden_size, device=device), types.view(1, 1, -1)], dim=2)


# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.reformat = nn.Linear(hidden_size + 4, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        if hidden.shape[2] > self.hidden_size:
            hidden = self.reformat(hidden)
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self, origin_hidden, types):
        return torch.cat([origin_hidden, types.view(1, 1, -1)], dim=2)
