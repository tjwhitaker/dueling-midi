import pretty_midi
import numpy as np
import pickle
import os
import torch


# Convert midi file to piano roll of the melody
def midi_to_roll(file):
    midi_data = pretty_midi.PrettyMIDI(file)

    for track in midi_data.instruments:
        if track.name == "MELODY":
            # Sixteenth notes
            roll = track.get_piano_roll(fs=track.get_end_time() / 16)
            trimmed = trim_roll(roll)

    return trimmed


# Remove empty space at the beginning and end of a piano roll
def trim_roll(roll):
    _, cols = roll.shape
    strip_indexes = []

    for i in range(cols):
        if np.sum(roll[:, i]) == 0.0:
            strip_indexes.append(i)
        else:
            break

    for i in range(cols):
        if np.sum(roll[:, (cols-1)-i]) == 0.0:
            strip_indexes.append(i)
        else:
            break

    return np.delete(roll, strip_indexes, axis=1)


# Convert piano roll to pitch and velocity sequences
def split_roll(roll):
    pitches = []
    velocities = []

    _, cols = roll.shape
    pitches = np.argmax(roll, axis=0)
    velocities = [int(roll[pitches[i], i]) for i in range(cols)]

    return pitches, velocities


# Build batch of (input, target) tuples from sequence
def sequence_to_batch(sequence, length):
    batch = []

    # Build batch by sliding a window by 1 space.
    # Final batch size ~ 3m tuples at 64 1/16 notes
    # for i in range(len(sequence) - length):
    #     inputs = sequence[i:i+length]
    #     targets = sequence[i+1:i+1+length]

    #     batch.append((inputs, targets))

    # Build batch by sliding the window by the sequence length
    # Final batch size ~45k at 64 1/16 notes
    ptr = 0

    while True:
        inputs = sequence[ptr: ptr+length]
        targets = sequence[ptr+1: ptr+1+length]

        batch.append((inputs, targets))

        ptr += length

        if ptr+length+1 > len(sequence):
            break

    return batch


def train(model, loader, epochs, device):
    total_loss = 0

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    for _ in range(epochs):
        for i, (inputs, targets) in enumerate(loader):
            hidden_state = model.init_hidden(inputs.size()[0])

            # (batch_size, sequence_length)
            inputs = inputs.to(device)
            targets = targets.to(device)

            output, hidden_state = model(inputs, hidden_state)

            # Loss function expects (batch_size, feature_dim, sequence_length)
            output = output.permute(0, 2, 1)

            loss = criterion(output, targets)
            total_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Loss: {total_loss}\n")


def generate_melody(model, initial_sequence, sequence_length, device):
    batch_size = 1

    hidden_state = model.init_hidden(batch_size)

    result = []

    for _ in range(sequence_length):
        output, hidden_state = model(initial_sequence, hidden_state)

        output = torch.functional.F.softmax(torch.squeeze(output), dim=0)
        dist = torch.distributions.Categorical(output)
        index = dist.sample()

        initial_sequence[0][0] = index.item()
        result.append(index.item())

    return result
