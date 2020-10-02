import torch
import torch.nn as nn


class MelodyLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, output_size=128):
        super(MelodyLSTM, self).__init__()

        self.embedding = nn.Embedding(input_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence, hidden_state):
        embedding = self.embedding(input_sequence)
        output, hidden_state = self.lstm(embedding, hidden_state)
        output = self.decoder(output)

        return output, hidden_state


class MelodyTransformer(nn.Module):
    def __init__(self):
        pass

    def forward(self):