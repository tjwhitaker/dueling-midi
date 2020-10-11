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

training_set = data.get_training_set(sequence_length)
input_sequence = torch.tensor(training_set[0][0]).to(device)


# TODO: get input sequence (rtmidi)

# Process melody into format for midi?
# Piano roll, or midi messages?

########################################
# TODO: Clean this up. Generate a melody.
batch_size = 1
hidden_state = model.init_hidden(batch_size)
melody = []

# Prime hidden state with input sequence
for note in input_sequence:
    note = note.unsqueeze(0).unsqueeze(0)
    output, hidden_state = model(note, hidden_state)

# Last predicted note from prime ^
output = torch.functional.F.softmax(output, dim=0)
predicted_note = torch.distributions.Categorical(output).sample()

input_note = torch.tensor([[predicted_note]]).to(device)

# Generate sequence
for _ in range(sequence_length):
    output, hidden_state = model(input_note, hidden_state)

    output = torch.functional.F.softmax(torch.squeeze(output), dim=0)
    note = torch.distributions.Categorical(output).sample()

    input_note[0][0] = note.item()
    melody.append(note.item())

print(melody)
##########################################

# TODO: Play send melody to midi device
