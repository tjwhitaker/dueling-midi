import pretty_midi
import numpy as np
import pickle
import os


# Load or build pickled dataset of processed midi files
def load_training_set():
    if os.path.exists("./dataset.pkl"):
        return pickle.load(open("dataset.pkl", "rb"))
    else:
        data = []

        for i in range(909):
            data.append(process_midi(f"./data/{(i+1):03}/{(i+1):03}.mid"))

        with open("dataset.pkl", "wb") as output:
            pickle.dump(data, output)

        return data


# Convert midi file to piano roll of the melody
def process_midi(file):
    midi_data = pretty_midi.PrettyMIDI(file)

    for track in midi_data.instruments:
        if track.name == "MELODY":
            # Sixteenth notes
            roll = track.get_piano_roll(fs=track.get_end_time() / 16)
            melody = trim_piano_roll(roll)

    return melody


# Remove empty space at the beginning and end of a piano roll
def trim_piano_roll(roll):
    rows, cols = roll.shape
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


# Process roll
def encode_roll():
    pass
