import numpy as np
import torch
from models import NoteLSTM
import utils
import rtmidi
import time

# sequence_length = 64
# num_pitches = 128
# hidden_size = 256
# hidden_layers = 2
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(device)

# # Load model
# model = NoteLSTM().to(device)

# model.load_state_dict(torch.load("../models/notelstm.model"))
# model.eval()

# training_set = utils.get_training_set(sequence_length)
# input_sequence = torch.tensor(training_set[0][0]).to(device)


# # TODO: get input sequence (rtmidi)

# # Process melody into format for midi?
# # Piano roll, or midi messages?

# ########################################
# # TODO: Clean this up. Generate a melody.
# batch_size = 1
# hidden_state = model.init_hidden(batch_size)
# melody = []

# # Prime hidden state with input sequence
# for note in input_sequence:
#     note = note.unsqueeze(0).unsqueeze(0)
#     output, hidden_state = model(note, hidden_state)

# # Last predicted note from prime ^
# output = torch.functional.F.softmax(output, dim=0)
# predicted_note = torch.distributions.Categorical(output).sample()

# input_note = torch.tensor([[predicted_note]]).to(device)

# # Generate sequence
# for _ in range(sequence_length):
#     output, hidden_state = model(input_note, hidden_state)

#     output = torch.functional.F.softmax(torch.squeeze(output), dim=0)
#     note = torch.distributions.Categorical(output).sample()

#     input_note[0][0] = note.item()
#     melody.append(note.item())

# print(input_sequence)
# print(melody)
# ##########################################

# # TODO: Send melody to midi device
# midiout = rtmidi.MidiOut()
# available_ports = midiout.get_ports()

# print(available_ports)

# if available_ports:
#     midiout.open_port(5)
# else:
#     midiout.open_virtual_port("My virtual output")

# with midiout:
#     for note in input_sequence:
#         note_on = [0x90, note, 112]
#         note_off = [0x80, note, 0]

#         if note == 0:
#             note_on = [0x90, note, 0]

#         midiout.send_message(note_on)
#         time.sleep(0.05)
#         midiout.send_message(note_off)

#     time.sleep(0.5)

#     for note in melody:
#         note_on = [0x90, note, 112]
#         note_off = [0x80, note, 0]

#         if note == 0:
#             note_on = [0x90, note, 0]

#         midiout.send_message(note_on)
#         time.sleep(0.05)
#         midiout.send_message(note_off)

# del midiout

from rtmidi.utils import open_midioutput, open_midiinput

if __name__ == "__main__":
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NoteLSTM().to(device)
    model.load_state_dict(torch.load("../models/notelstm.model"))
    model.eval()

    # Open midi ports/callbacks
    midi_port = 5

    try:
        midi_in, port_name = open_midiinput(midi_port)
        midi_out, port_name = open_midioutput(midi_port)

        midi_in.set_callback(MidiInputHandler(port_name))

    except (EOFError, KeyboardInterrupt):
        print("Exit.")

    finally:
        midi_in.close_port()
        midi_out.close_port()

        del midi_in, midi_out

    print("Attaching MIDI input callback handler.")
    midiin.set_callback(MidiInputHandler(port_name))

    print("Entering main loop. Press Control-C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('')
    finally:
        print("Exit.")
        midiin.close_port()
        del midiin

    open_midi_input(model)
    open_midi_output()
