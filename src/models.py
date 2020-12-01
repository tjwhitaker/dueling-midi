import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        # 128 Valid Midi Notes
        self.input_size = 128
        self.output_size = 128

        # LSTM Hyperparams
        self.hidden_size = 256
        self.hidden_layers = 1
        self.embedding_dimensions = 4

        # Network Structure
        self.embedding = nn.Embedding(
            self.input_size, self.embedding_dimensions)

        self.lstm = nn.LSTM(self.embedding_dimensions, self.hidden_size,
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


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        # 128 Valid Midi Notes
        self.input_size = 128
        self.output_size = 128

        # GRU Hyperparams
        self.hidden_size = 256
        self.hidden_layers = 1
        self.embedding_dimensions = 4

        # Network Structure
        self.embedding = nn.Embedding(
            self.input_size, self.embedding_dimensions)

        self.gru = nn.GRU(self.embedding_dimensions, self.hidden_size,
                          self.hidden_layers, batch_first=True)

        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden_state):
        embedding = self.embedding(x)
        output, hidden_state = self.gru(embedding, hidden_state)
        logits = self.decoder(output)

        return logits, hidden_state

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_size)).cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.sequence_length = 32

        # 128 Valid Midi Notes
        self.input_size = 128
        self.output_size = 128

        # CNN Hyperparams
        self.embedding_dimensions = 32

        # Network Structure
        self.embedding = nn.Embedding(
            self.input_size, self.embedding_dimensions)

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=self.sequence_length,
                      out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=32,
                      out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.decoder = nn.Linear(896*2, self.output_size)

    def forward(self, x):
        embedding = self.embedding(x)
        features = self.conv_block(embedding)
        logits = self.decoder(features)

        return logits


class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()

        # 128 Valid Midi Notes
        self.input_size = 128
        self.output_size = 128

        # LSTM Hyperparams
        self.hidden_size = 128
        self.hidden_layers = 1
        self.embedding_dimensions = 32

        # Network Structure
        self.embedding = nn.Embedding(
            self.input_size, self.embedding_dimensions)

        self.lstm = nn.LSTM(self.embedding_dimensions, self.hidden_size,
                            self.hidden_layers, batch_first=True)

    def forward(self, x, hidden_state):
        embedding = self.embedding(x)
        output, hidden_state = self.lstm(embedding, hidden_state)

        return output, hidden_state

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_size)).cuda(),
                Variable((torch.zeros(self.hidden_layers, batch_size, self.hidden_size)).cuda()))


class LSTMDecoder(nn.Module):
    def __init__(self):
        super(LSTMDecoder, self).__init__()

        # 128 Valid Midi Notes
        self.input_size = 128
        self.output_size = 128

        # LSTM Hyperparams
        self.hidden_size = 128
        self.hidden_layers = 1
        self.embedding_dimensions = 32

        # Network Structure
        self.embedding = nn.Embedding(
            self.input_size, self.embedding_dimensions)

        self.lstm = nn.LSTM(self.embedding_dimensions, self.hidden_size,
                            self.hidden_layers, batch_first=True)

        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden_state):
        embedding = self.embedding(x)
        output, hidden_state = self.lstm(embedding, hidden_state)
        logits = self.decoder(output)

        return logits, hidden_state


class GRUEncoder(nn.Module):
    def __init__(self):
        super(GRUEncoder, self).__init__()

        # 128 Valid Midi Notes
        self.input_size = 128
        self.output_size = 128

        # LSTM Hyperparams
        self.hidden_size = 128
        self.hidden_layers = 1
        self.embedding_dimensions = 32

        # Network Structure
        self.embedding = nn.Embedding(
            self.input_size, self.embedding_dimensions)

        self.gru = nn.GRU(self.embedding_dimensions, self.hidden_size,
                          self.hidden_layers, batch_first=True)

    def forward(self, x, hidden_state):
        embedding = self.embedding(x)
        output, hidden_state = self.gru(embedding, hidden_state)

        return output, hidden_state

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_size)).cuda()


class GRUDecoder(nn.Module):
    def __init__(self):
        super(GRUDecoder, self).__init__()

        # 128 Valid Midi Notes
        self.input_size = 128
        self.output_size = 128

        # LSTM Hyperparams
        self.hidden_size = 128
        self.hidden_layers = 1
        self.embedding_dimensions = 32

        # Network Structure
        self.embedding = nn.Embedding(
            self.input_size, self.embedding_dimensions)

        self.gru = nn.GRU(self.embedding_dimensions, self.hidden_size,
                          self.hidden_layers, batch_first=True)

        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden_state):
        embedding = self.embedding(x)
        output, hidden_state = self.gru(embedding, hidden_state)
        logits = self.decoder(output)

        return logits, hidden_state
