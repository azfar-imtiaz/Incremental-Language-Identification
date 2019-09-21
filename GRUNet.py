import torch.nn as nn


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax()

    def forward(self, sequence):
        hidden_state = self.init_hidden(len(sequence))
        output, hidden = self.gru(sequence, hidden_state)
        # output = output.contiguous().view(-1, len(sentence))
        output = self.linear(output)
        output = self.softmax(output)
        return output

    def init_hidden(self, sequence_length):
        return torch.zeros(sequence_length, self.hidden_size)
