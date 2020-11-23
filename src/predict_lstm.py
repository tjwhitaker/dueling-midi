import torch
from models import NoteLSTM
import utils


def predict_lstm(model, input_sequence, sequence_length=64):
    hidden_state = model.init_hidden(input_sequence.shape[0])
    melody = []

    # Prime hidden state with input sequence
    for note in input_sequence[0]:
        note = note.unsqueeze(0).unsqueeze(0)
        output, hidden_state = model(note, hidden_state)

    # Last predicted note from prime ^
    output = torch.functional.F.softmax(output, dim=0)
    predicted_note = torch.distributions.Categorical(output).sample()

    # Shift sequence and replace oldest note with prediction
    input_sequence = torch.roll(input_sequence, shifts=-1, dims=1)
    input_sequence[0][-1] = predicted_note.item()

    # Generate sequence
    for _ in range(sequence_length):
        # Pass most recent note to model
        input_note = input_sequence[0][-1].unsqueeze(0).unsqueeze(0)
        output, hidden_state = model(input_note, hidden_state)

        output = torch.functional.F.softmax(torch.squeeze(output), dim=0)
        note = torch.distributions.Categorical(output).sample()

        # Shift sequence and replace oldest note with prediction
        input_sequence = torch.roll(input_sequence, shifts=-1, dims=1)
        input_sequence[0][-1] = note.item()
        melody.append(note.item())

    return melody


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NoteLSTM().to(device)
    model.load_state_dict(torch.load("../models/notelstm.model"))
    model.eval()

    # input_sequence = torch.tensor([[61, 61,  0, 63,  0, 66,  0,  0, 68,  0,  0, 70,  0,  0,  0,  0, 66, 66, 66,  0,  0, 63, 63, 63,
    #                                 63,  0,  0, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  0,  0,  0,  0,  0,
    #                                 0,  0,  0,  0,  0, 68, 68, 68, 68,  0,  0, 65, 65, 65,  0,  0]]).to(device)

    training_set = utils.get_training_set(sequence_length=32)
    input_sequence = torch.tensor(training_set[0][0]).unsqueeze(0).to(device)

    melody = predict_lstm(model, input_sequence, sequence_length=64)

    print(input_sequence)
    print(melody)
