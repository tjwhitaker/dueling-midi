import pretty_midi
import numpy as np
import pickle
import os
import torch


# Expects data to be a list of (input, target) tuples.
class MelodyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Convert midi file to piano roll of the melody
def midi_to_roll(file):
    midi_data = pretty_midi.PrettyMIDI(file)

    for track in midi_data.instruments:
        if track.name == "MELODY":
            roll = track.get_piano_roll(fs=16)
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


# Used to augment dataset
# Return a list of melodies
def transpose_melody(melody, octaves):
    high = 108
    low = 21
    melodies = [melody]
    melody_min = np.min(melody[np.nonzero(melody)])

    for i in range(octaves):
        shift = (i+1)*12

        if (melody_min - shift) >= low:
            transposed = []

            for note in melody:
                if note == 0:
                    transposed.append(0)
                else:
                    transposed.append(note-shift)

            melodies.append(np.array(transposed))

        if (max(melody) + shift) <= high:
            transposed = []

            for note in melody:
                if note == 0:
                    transposed.append(0)
                else:
                    transposed.append(note+shift)

            melodies.append(np.array(transposed))

    return melodies


# Build batch of (input, target) tuples from sequence
def sequence_to_batch(sequence, length):
    batch = []

    # Build batch by sliding a window by 1 space.
    # Final batch size ~ 3m tuples at 64 1/16 notes
    # for i in range(len(sequence) - length):
    #     inputs = sequence[i:i+length]
    #     targets = sequence[i+1:i+1+length]

    #     batch.append((inputs, targets))

    # Build batch by sliding a window by the sequence length
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


# Load or build pickled dataset of processed midi files
def get_training_set(sequence_length=32):
    if os.path.exists("../data/processed/melody_dataset.pkl"):
        return pickle.load(open("../data/processed/melody_dataset.pkl", "rb"))
    else:
        rolls = []

        for i in range(909):
            roll = midi_to_roll(
                f"../data/external/pop909/{(i+1):03}/{(i+1):03}.mid")
            rolls.append(roll)

        batches = []

        print(len(rolls))

        for roll in rolls:
            pitches, _ = split_roll(roll)

            melodies = transpose_melody(pitches, 3)

            for melody in melodies:
                batch = sequence_to_batch(melody, sequence_length)
                batches.append(batch)

        # Flatten list of lists of tuples to list of tuples
        # Size of data: torch.Size([2855383, 2, 64])
        data = [item for sublist in batches for item in sublist]
        training_set = MelodyDataset(data)

        with open('../data/processed/melody_dataset.pkl', 'wb') as output:
            pickle.dump(training_set, output)

        return training_set
