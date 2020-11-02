import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NoteLSTM(nn.Module):
    def __init__(self, input_size=128, output_size=128, hidden_size=256, hidden_layers=2):
        super(NoteLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.embedding = nn.Embedding(input_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_size,
                            hidden_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        embedding = self.embedding(x)
        output, hidden_state = self.lstm(embedding, hidden_state)
        logits = self.decoder(output)

        return logits, hidden_state

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_size)).cuda(),
                Variable((torch.zeros(self.hidden_layers, batch_size, self.hidden_size)).cuda()))


class NoteCNN(nn.Module):
    def __init__(self, input_size=128, output_size=128, self.sequence_length):
        in_channels = 1
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length

        self.embedding = nn.Embedding(input_size, output_size)

        self.conv_layers = nn.Sequential([
            nn.Conv1d(in_channels, 16, kernel_size=8),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.ReLU()
        ])

        self.decoder = nn.Sequential([
            nn.Linear(32*3*3, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        ])

    def forward(self, x):
        embedding = self.embedding(x)
        features = self.conv_layers(embedding)
        logits = self.decoder(features)

        return logits


class SequenceLSTM(nn.Module):
    def __init__(self):
        pass


class SequenceCNN(nn.Module):
    def __init__(self):
        pass
