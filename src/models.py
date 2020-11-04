import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NoteLSTM(nn.Module):
    def __init__(self):
        super(NoteLSTM, self).__init__()

        # 128 Valid Midi Notes
        self.input_size = 128
        self.output_size = 128

        # LSTM Hyperparams
        self.hidden_size = 256
        self.hidden_layers = 2
        self.embedding_dimensions = 32

        # Network Structure
        self.embedding = nn.Embedding(
            self.input_size, self.embedding_dimensions)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            self.hidden_layers, batch_first=True)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden_state):
        embedding = self.embedding(x)
        output, hidden_state = self.lstm(embedding, hidden_state)
        logits = self.decoder(output)

        return logits, hidden_state

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_size)).cuda(),
                Variable((torch.zeros(self.hidden_layers, batch_size, self.hidden_size)).cuda()))


class NoteCNN(nn.Module):
    def __init__(self):
        super(NoteCNN, self).__init__()

        # 128 Valid Midi Notes
        self.input_size = 128
        self.output_size = 128

        # CNN Hyperparams
        self.embedding_dimensions = 32

        # Network Structure
        self.embedding = nn.Embedding(
            self.input_size, self.embedding_dimensions)

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=8),
            nn.ReLU(),
            nn.Flatten()
        )

        self.decoder = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size)
        )

    def forward(self, x):
        embedding = self.embedding(x)
        features = self.conv_block(embedding)
        logits = self.decoder(features)

        return logits
