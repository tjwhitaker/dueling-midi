import torch
from models import NoteCNN
import utils


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

    model = NoteCNN().to(device)
    model.load_state_dict(torch.load("../models/notecnn.model"))
    model.eval()

    input_sequence = torch.tensor([[61, 61,  0, 63,  0, 66,  0,  0, 68,  0,  0, 70,  0,  0,  0,  0, 66, 66, 66,  0,  0, 63, 63, 63,
                                    63,  0,  0, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0, 68, 68, 68, 68,  0,  0, 65, 65, 65,  0,  0]]).to(device)

    melody = predict_cnn(model, input_sequence)

    print(melody)
