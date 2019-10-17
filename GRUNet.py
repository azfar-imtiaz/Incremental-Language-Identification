import torch.nn as nn
import torch

import config


class GRUNet(nn.Module):
    def __init__(self, vocab_size, seq_len, input_size, hidden_size, num_layers, output_size, dev, dropout=0.0):
        super().__init__()
        self.num_layers = config.GRU_NUM_LAYERS
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dev = dev
        self.emb = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * seq_len, output_size)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, sequence, hidden_layer):
        output = self.emb(sequence)
        hidden_layer = hidden_layer.to(self.dev)
        output, hidden_layer = self.gru(output, hidden_layer)
        output = output.contiguous().view(-1, self.hidden_size *
                                          len(sequence[0]))
        output = self.fc(output)
        # don't need the softmax here as CrossEntropy loss already does softmax at its end
        # output = self.softmax(output)
        return output, hidden_layer

    def init_hidden(self, seq_len):
        return torch.zeros(self.num_layers, seq_len, self.hidden_size).float()


class CharMinimizationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, num_chars):
        output = self.fc(torch.FloatTensor([num_chars]))
        return output
