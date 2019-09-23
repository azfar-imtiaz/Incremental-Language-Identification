import torch.nn as nn
import torch.nn.functional as F
import torch


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, dropout=self.dropout, batch_first=False)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence):
        hidden_state = self.init_hidden(len(sequence))
        # print(hidden_state.type())
        # print(sequence.type())
        output, hidden = self.gru(sequence, hidden_state)
        # output = output.contiguous().view(-1, len(sequence))
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = self.softmax(output)
        return output

    def init_hidden(self, sequence_length):
        # return torch.Tensor([
        #     torch.zeros(self.num_layers, sequence_length,
        #                 self.hidden_size, dtype=torch.float),
        #     torch.zeros(self.num_layers, sequence_length,
        #                 self.hidden_size, dtype=torch.float),
        #     torch.zeros(self.num_layers, sequence_length,
        #                 self.hidden_size, dtype=torch.float)
        # ])
        # print("Hidden layer size: ")
        # print(torch.randn([self.batch_size, sequence_length,
        #                    self.hidden_size], dtype=torch.float).size())
        return torch.randn([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float)
