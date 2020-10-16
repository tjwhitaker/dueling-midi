import torch
import torch.nn as nn
from torch.autograd import Variable


class MelodyLSTM(nn.Module):
    def __init__(self, input_size=128, output_size=128, hidden_size=256, hidden_layers=2):
        super(MelodyLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.embedding = nn.Embedding(input_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size,
                            hidden_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence, hidden_state):
        embedding = self.embedding(input_sequence)
        output, hidden_state = self.lstm(embedding, hidden_state)
        output = self.decoder(output)

        return output, hidden_state

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_size)).cuda(),
                Variable((torch.zeros(self.hidden_layers, batch_size, self.hidden_size)).cuda()))
