import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reparameter(mu, log_var):
    std = torch.exp(log_var * 0.5)
    esp = torch.randn_like(std)
    return mu + std * esp


# Encoder
# TODO: apply LSTM
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.reformat_hidden = nn.Linear(in_features=hidden_size + 4, out_features=hidden_size)
        self.reformat_cell = nn.Linear(in_features=hidden_size + 4, out_features=hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.mean_hidden = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.logvar_hidden = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mean_cell = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.logvar_cell = nn.Linear(in_features=hidden_size, out_features=hidden_size)

    def forward(self, input, hidden, cell):
        # Reformat and word embedding
        hidden = self.reformat_hidden(hidden).view(1, 1, -1)
        cell = self.reformat_cell(cell).view(1, 1, -1)
        embedded = self.embedding(input).view(len(input), 1, -1)  # len(input) x hidden_size

        # Do the rnn part
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # Compute the mean and variance
        mean_h, logvar_h = self.mean_hidden(hidden), self.logvar_hidden(hidden)
        mean_c, logvar_c = self.mean_cell(cell), self.logvar_cell(cell)
        return (mean_h, logvar_h), (mean_c, logvar_c)

    def initHidden(self, types):
        return torch.cat([torch.zeros(1, 1, self.hidden_size, device=device), types.view(1, 1, -1)], dim=2)


# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.reformat = nn.Linear(hidden_size + 4, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        if hidden.shape[2] > self.hidden_size:
            hidden = self.reformat(hidden).view(1, 1, -1)
        if cell.shape[2] > self.hidden_size:
            cell = self.reformat(cell).view(1, 1, -1)
        input = self.embedding(input).view(1, 1, -1)
        input = F.relu(input)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = self.out(output[0])
        return output, hidden, cell

    def initHidden(self, origin_hidden, types):
        return torch.cat([origin_hidden, types.view(1, 1, -1)], dim=2)
