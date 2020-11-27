from predict_lstm import predict_lstm
from predict_seq2seq import predict_seq2seq
from predict_gru import predict_gru
from predict_cnn import predict_cnn
from models import NoteLSTM, NoteGRU, NoteCNN, Encoder, Decoder
import torch
import utils
import time
import pretty_midi
import mido

# Params for midi streaming
port_name = "USB Midi:USB Midi MIDI 1 28:0"
note_buffer = []
start_time = time.time()
wallclock = 0
note_start = 0

# Pretty midi for piano roll
# Probably a good place to improve performance.
# TODO: Better way to generate sequence of notes.
instrument = pretty_midi.Instrument(2)

print("Setting up the models")

# Setting up the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm = NoteLSTM().to(device)
lstm.load_state_dict(torch.load("../models/notelstm.model"))
lstm.eval()

gru = NoteGRU().to(device)
gru.load_state_dict(torch.load("../models/notegru.model"))
gru.eval()

cnn = NoteCNN().to(device)
cnn.load_state_dict(torch.load("../models/notecnn.model"))
cnn.eval()

encoder = Encoder().to(device)
decoder = Decoder().to(device)

encoder.load_state_dict(torch.load("../models/encoder.model"))
decoder.load_state_dict(torch.load("../models/decoder.model"))

encoder.eval()
decoder.eval()

print("Opening midi ports")

triggers = [105, 106, 107, 108]

with mido.open_output(port_name) as outport:
    with mido.open_input(port_name) as inport:
        for msg in inport:
            print(msg)

            wallclock = time.time() - start_time

            if msg.type == "note_on":
                note_start = wallclock
            if msg.type == "note_off":
                # Trigger neural network
                if msg.note in triggers:
                    # Build input
                    print("Getting conditional input")
                    roll = instrument.get_piano_roll(fs=16)
                    trimmed = utils.trim_roll(roll)
                    pitches, _ = utils.split_roll(trimmed)
                    input_sequence = torch.tensor([pitches[-32:]]).to(device)

                    print(input_sequence)

                    # Generate melody
                    print("Generating melody")
                    if msg.note == 105:
                        melody = predict_cnn(
                            cnn, input_sequence, sequence_length=64)
                    if msg.note == 106:
                        melody = predict_lstm(
                            lstm, input_sequence, sequence_length=64)
                    if msg.note == 107:
                        melody = predict_gru(
                            gru, input_sequence, sequence_length=64)
                    if msg.note == 108:
                        melody = predict_seq2seq(
                            encoder, decoder, input_sequence, sequence_length=64)

                    print(melody)

                    # Play melody
                    previous_note = None

                    for note in melody:
                        if previous_note == None:
                            if note != 0:
                                outport.send(mido.Message(
                                    type="note_on", note=note))

                        elif note != previous_note:
                            if previous_note != 0:
                                outport.send(mido.Message(
                                    type="note_off", note=previous_note))

                            if note != 0:
                                outport.send(mido.Message(
                                    type="note_on", note=note))

                        previous_note = note
                        time.sleep(1./16)

                    outport.send(mido.Message(
                        type="note_off", note=previous_note))

                    # Clear buffer
                    print("Clearing note buffer")
                    instrument.notes = []

                else:
                    note_end = wallclock
                    note = pretty_midi.Note(
                        start=note_start, end=note_end, pitch=msg.note, velocity=100)
                    instrument.notes.append(note)
