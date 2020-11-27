import torch
from models import Encoder, Decoder
import utils


def predict_seq2seq(encoder, decoder, input_sequence, sequence_length=64):
    hidden_state = encoder.init_hidden(input_sequence.shape[0])
    melody = []

    # Prime hidden state with input sequence
    for note in input_sequence[0]:
        note = note.unsqueeze(0).unsqueeze(0)
        encoder_output, hidden_state = encoder(note, hidden_state)

    # Use input sequence tensor to avoid allocating another tensor
    input_sequence[0][0] = 0

    # Generate sequence
    for _ in range(sequence_length):
        # Pass most recent note to model
        decoder_output, hidden_state = decoder(
            input_sequence[0][0].unsqueeze(0).unsqueeze(0), hidden_state)

        output = torch.functional.F.softmax(
            torch.squeeze(decoder_output), dim=0)
        note = torch.distributions.Categorical(output).sample()

        input_sequence[0][0] = note.item()
        melody.append(note.item())

    return melody


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    encoder.load_state_dict(torch.load("../models/encoder.model"))
    decoder.load_state_dict(torch.load("../models/decoder.model"))

    encoder.eval()
    decoder.eval()

    input_sequence = torch.tensor([[61, 61,  0, 63,  0, 66,  0,  0, 68,  0,  0, 70,  0,  0,  0,  0, 66, 66, 66,  0,  0, 63, 63, 63,
                                    63,  0,  0, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0, 68, 68, 68, 68,  0,  0, 65, 65, 65,  0,  0]]).to(device)

    melody = predict_seq2seq(encoder, decoder, input_sequence)

    print(melody)
    print(len(melody))
