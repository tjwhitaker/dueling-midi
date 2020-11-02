import torch
import torch.nn as nn
from torch.autograd import Variable


class Melody2Melody(nn.Module):
    def __init__(self):
        pass


class EncoderLSTM(nn.Module):
    def __init__(self, input_size=128, output_size=128, hidden_size=256, hidden_layers=2):
        pass

    def forward(self, input_sequence, hidden_state):
        pass

    def init_hidden(self, batch_size):
        pass


class DecoderLSTM(nn.Module):
    def __init__(self, input_size=128, output_size=128, hidden_size=256, hidden_layers=2):
        pass

    def forward(self, input_sequence, hidden_state):
        pass

    def init_hidden(self, batch_size):
        pass
