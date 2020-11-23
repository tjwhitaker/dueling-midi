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


class NoteGRU(nn.Module):
    def __init__(self):
        super(NoteGRU, self).__init__()

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


class NoteCNN(nn.Module):
    def __init__(self):
        super(NoteCNN, self).__init__()

        self.sequence_length = 64

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
                      out_channels=16, kernel_size=2),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.decoder = nn.Sequential(
            nn.Linear(240, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

    def forward(self, x):
        embedding = self.embedding(x)
        features = self.conv_block(embedding)
        logits = self.decoder(features)

        return logits


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

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


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

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


# class Seq2Seq(nn.Module):
#     def __init__(self):
#         super(Seq2Seq, self).__init__()

#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def forward(self, x, hidden_state):
#         encoder_output, hidden_state = self.encoder(x, hidden_state)
#         decoder_output, hidden_state = self.decoder(x, hidden_state)

#         return decoder_output, hidden_state

#     def init_hidden(self, batch_size):
#         return self.encoder.init_hidden(batch_size)
