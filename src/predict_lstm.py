import torch
from models import NoteLSTM
import utils


def predict_lstm(model, device, input_sequence, sequence_length=64):
    hidden_state = model.init_hidden(input_sequence.shape[0])
    melody = []

    # Prime hidden state with input sequence
    for note in input_sequence[0]:
        note = note.unsqueeze(0).unsqueeze(0)
        output, hidden_state = model(note, hidden_state)

    # Last predicted note from prime ^
    output = torch.functional.F.softmax(output, dim=0)
    predicted_note = torch.distributions.Categorical(output).sample()

    input_note = torch.tensor([[predicted_note]]).to(device)

    # Generate sequence
    for _ in range(sequence_length):
        output, hidden_state = model(input_note, hidden_state)

        output = torch.functional.F.softmax(torch.squeeze(output), dim=0)
        note = torch.distributions.Categorical(output).sample()

        input_note[0][0] = note.item()
        melody.append(note.item())

    return melody


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NoteLSTM().to(device)
    model.load_state_dict(torch.load("../models/notelstm.model"))
    model.eval()

    input_sequence = torch.tensor([[61, 61,  0, 63,  0, 66,  0,  0, 68,  0,  0, 70,  0,  0,  0,  0, 66, 66, 66,  0,  0, 63, 63, 63,
                                    63,  0,  0, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0, 68, 68, 68, 68,  0,  0, 65, 65, 65,  0,  0]]).to(device)

    melody = predict_lstm(model, device, input_sequence)

    print(melody)
