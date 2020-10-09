import numpy as np
import torch
import models
import utils


# Expects data to be a list of (input, target) tuples.
def train(model, data):
    total_loss = 0

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Batch size of 1
    hidden_state = model.init_hidden(1)

    for inputs, targets in data:
        # Add batch dimension (sequence_length, batch_size, feature_dim)
        inputs = torch.unsqueeze(inputs, dim=1)
        targets = torch.unsqueeze(targets, dim=1)

        hidden_state = tuple([each.data for each in hidden_state])

        output, hidden_state = model(inputs, hidden_state)

        loss = criterion(torch.squeeze(output), torch.squeeze(targets))
        total_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {total_loss}\n")


def generate_sample(model, initial_sequence):
    sequence_length = 64
    batch_size = 1

    hidden_state = model.init_hidden(batch_size)

    # input_seq = torch.tensor([[66]]).to(device)
    result = []

    for _ in range(sequence_length):
        output, hidden_state = model(initial_sequence, hidden_state)

        output = torch.functional.F.softmax(torch.squeeze(output), dim=0)
        dist = torch.distributions.Categorical(output)
        index = dist.sample()

        initial_sequence[0][0] = index.item()
        result.append(index.item())

    return result


epochs = 1
sequence_length = 64
batch_size = 1
num_pitches = 128
hidden_size = 256
hidden_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and process training data.
# Batches = [[(inputs, targets), ...], ...]
rolls = utils.load_rolls()
batches = []

for roll in rolls:
    pitches, velocities = utils.split_roll(roll)
    batch = utils.sequence_to_batch(pitches, sequence_length)
    batches.append(batch)

# Flatten list of lists of tuples to list of tuples
# Size of data: torch.Size([2855383, 2, 64])
data = torch.tensor(
    [item for sublist in batches for item in sublist]).to(device)

print(data.size())

# data = batches[0].to(device)

model = models.MelodyLSTM(num_pitches, num_pitches,
                          hidden_size, hidden_layers).to(device)

for i in range(epochs):
    print(f"Epoch {i}")
    print("---------------------------")
    train(model, data)

# Save model
torch.save(model.state_dict(), "MelodyLSTM.model")

# Load model just to test
model = models.MelodyLSTM(num_pitches, num_pitches,
                          hidden_size, hidden_layers).to(device)

model.load_state_dict(torch.load("MelodyLSTM.model"))
model.eval()

initial_sequence = torch.tensor([[66]]).to(device)
melody = generate_sample(model, initial_sequence)
print(melody)
