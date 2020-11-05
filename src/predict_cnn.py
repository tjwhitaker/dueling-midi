import torch
from models import NoteCNN
import utils

sequence_length = 64
num_pitches = 128

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# Load model
model = NoteCNN().to(device)

model.load_state_dict(torch.load("../models/notecnn.model"))
model.eval()

training_set = utils.get_training_set(sequence_length)
input_sequence = torch.tensor(training_set[0][0]).unsqueeze(0).to(device)
melody = []

# Generate sequence
for _ in range(sequence_length):
    output = model(input_sequence)

    output = torch.functional.F.softmax(torch.squeeze(output), dim=0)
    note = torch.distributions.Categorical(output).sample()

    # Shift sequence and replace oldest note with prediction
    input_sequence = torch.roll(input_sequence, shifts=-1, dims=1)
    input_sequence[0][-1] = note.item()

    melody.append(note.item())

print(training_set[0][0])
print(melody)
