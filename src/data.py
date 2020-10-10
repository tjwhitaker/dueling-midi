import pickle
import utils
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


def get_training_set(sequence_length):
    # Load or build pickled dataset of processed midi files
    if os.path.exists("./melody_dataset.pkl"):
        return pickle.load(open("melody_dataset.pkl", "rb"))
    else:
        rolls = []

        for i in range(909):
            roll = utils.midi_to_roll(f"./data/{(i+1):03}/{(i+1):03}.mid")
            rolls.append(roll)

        batches = []

        for roll in rolls:
            pitches, velocities = utils.split_roll(roll)
            batch = utils.sequence_to_batch(pitches, sequence_length)
            batches.append(batch)

        # Flatten list of lists of tuples to list of tuples
        # Size of data: torch.Size([2855383, 2, 64])
        data = [item for sublist in batches for item in sublist]
        training_set = MelodyDataset(data)

        with open('melody_dataset.pkl', 'wb') as output:
            pickle.dump(training_set, output)

        return training_set
