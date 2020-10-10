import numpy as np
import torch
import models
import utils
import data

sequence_length = 64
num_pitches = 128
hidden_size = 256
hidden_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.MelodyLSTM(num_pitches, num_pitches,
                          hidden_size, hidden_layers).to(device)

model.load_state_dict(torch.load("MelodyLSTM.model"))
model.eval()

# TODO: get input sequence (rtmidi)


inputs = torch.tensor([[66]]).to(device)

# Process melody into format for midi?
# Piano roll, or midi messages?
melody = utils.generate_melody(model, inputs, sequence_length, device)
print(melody)

# TODO: Play send melody to midi device
