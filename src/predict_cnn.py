import torch
from models import CNN
import utils
from time import time


def predict_cnn(model, input_sequence, sequence_length=64):
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

    return melody


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)
    model.load_state_dict(torch.load("../models/cnn.model"))
    model.eval()

    input_sequence = torch.tensor([[61, 61,  0, 63,  0, 66,  0,  0, 68,  0,  0, 70,  0,  0,  0,  0, 66, 66, 66,  0,  0, 63, 63, 63,
                                    63,  0,  0, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0, 68, 68, 68, 68,  0,  0, 65, 65, 65,  0,  0]]).to(device)

    training_set = utils.get_training_set(sequence_length=32)
    input_sequence = torch.tensor(training_set[0][0]).unsqueeze(0).to(device)

    times = []
    for i in range(100):
        start = time()
        melody = predict_cnn(model, input_sequence)
        times.append(time() - start)

    print(times)
