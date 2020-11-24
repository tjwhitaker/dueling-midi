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

print("Setting up the model")

# Setting up the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = NoteLSTM().to(device)
# model.load_state_dict(torch.load("../models/notelstm.model"))
# model.eval()

# model = NoteGRU().to(device)
# model.load_state_dict(torch.load("../models/notegru.model"))
# model.eval()

model = NoteCNN().to(device)
model.load_state_dict(torch.load("../models/notecnn.model"))
model.eval()

# encoder = Encoder().to(device)
# decoder = Decoder().to(device)

# encoder.load_state_dict(torch.load("../models/encoder.model"))
# decoder.load_state_dict(torch.load("../models/decoder.model"))

# encoder.eval()
# decoder.eval()

print("Opening midi ports")

with mido.open_output(port_name) as outport:
    with mido.open_input(port_name) as inport:
        for msg in inport:
            print(msg)

            wallclock = time.time() - start_time

            if msg.type == "note_on":
                note_start = wallclock
            if msg.type == "note_off":
                # Trigger neural network
                if msg.note == 108:
                    # Generate Melody
                    print("Getting conditional input")
                    roll = instrument.get_piano_roll(fs=16)
                    trimmed = utils.trim_roll(roll)
                    pitches, _ = utils.split_roll(trimmed)

                    print(pitches[-32:])

                    print("Generating melody")
                    input_sequence = torch.tensor([pitches[-32:]]).to(device)
                    melody = predict_cnn(
                        model, input_sequence, sequence_length=64)

                    # melody = predict_seq2seq(
                    #     encoder, decoder, input_sequence, sequence_length=64)

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
