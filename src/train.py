import numpy as np
import torch
import models
import utils


def train(model, device):
    epochs = 10
    sequence_length = 64
    batch_size = 1

    # Load and process training data.
    # Batches = [[(inputs, targets), ...], ...]
    rolls = utils.load_rolls()
    batches = []

    for roll in rolls:
        pitches, velocities = utils.split_roll(roll)
        batch = torch.tensor(utils.sequence_to_batch(pitches, sequence_length))
        batches.append(batch)

    # Just train on one song for now
    data = batches[0].to(device)

    # Prepare model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Training loop
    for i in range(epochs):
        total_loss = 0
        hidden_state = model.init_hidden(batch_size)

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

        print(f"Epoch: {i} \t Loss: {total_loss}")

        generate_sample(model)


def generate_sample(model):
    sequence_length = 64
    batch_size = 1

    hidden_state = model.init_hidden(batch_size)

    input_seq = torch.tensor([[66]]).to(device)

    print(input_seq.item(), end=' ')

    for _ in range(sequence_length):
        output, hidden_state = model(input_seq, hidden_state)

        output = torch.functional.F.softmax(torch.squeeze(output), dim=0)
        dist = torch.distributions.Categorical(output)
        index = dist.sample()

        print(index.item(), end=' ')

        input_seq[0][0] = index.item()

    print("\n")


num_pitches = 128
hidden_size = 256
hidden_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.MelodyLSTM(num_pitches, num_pitches,
                          hidden_size, hidden_layers).to(device)

train(model, device)
