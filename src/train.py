import numpy as np
import torch
import models
import utils
import data

epochs = 1
sequence_length = 64
batch_size = 64
num_pitches = 128
hidden_size = 256
hidden_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_set = data.get_training_set(sequence_length)
loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)

model = models.MelodyLSTM(num_pitches, num_pitches,
                          hidden_size, hidden_layers).to(device)

utils.train(model, loader, epochs, device)

# Save model
torch.save(model.state_dict(), "MelodyLSTM.model")
