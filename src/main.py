from predict_lstm import predict_lstm
from predict_gru import predict_gru
from predict_cnn import predict_cnn
from predict_lstm_enc_dec import predict_lstm_enc_dec
from predict_gru_enc_dec import predict_gru_enc_dec
from models import LSTM, GRU, CNN, LSTMEncoder, LSTMDecoder, GRUEncoder, GRUDecoder
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

lstm = LSTM().to(device)
lstm.load_state_dict(torch.load("../models/lstm.model"))
lstm.eval()

gru = GRU().to(device)
gru.load_state_dict(torch.load("../models/gru.model"))
gru.eval()

cnn = CNN().to(device)
cnn.load_state_dict(torch.load("../models/cnn.model"))
cnn.eval()

lstmencoder = LSTMEncoder().to(device)
lstmdecoder = LSTMDecoder().to(device)

lstmencoder.load_state_dict(torch.load("../models/lstmencoder.model"))
lstmdecoder.load_state_dict(torch.load("../models/lstmdecoder.model"))

lstmencoder.eval()
lstmdecoder.eval()

gruencoder = GRUEncoder().to(device)
grudecoder = GRUDecoder().to(device)

gruencoder.load_state_dict(torch.load("../models/gruencoder.model"))
grudecoder.load_state_dict(torch.load("../models/grudecoder.model"))

gruencoder.eval()
grudecoder.eval()

print("Opening midi ports")

triggers = [101, 103, 105, 107, 108]

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
                    if msg.note == 101:
                        print("Using CNN")
                        if len(pitches[-32:]) == 32:
                            input_sequence = torch.tensor(
                                [pitches[-32:]]).to(device)
                            melody = predict_cnn(
                                cnn, input_sequence, sequence_length=64)
                        else:
                            print("Not enough input for cnn")
                    if msg.note == 103:
                        print("Using LSTM")
                        melody = predict_lstm(
                            lstm, input_sequence, sequence_length=64)
                    if msg.note == 105:
                        print("Using GRU")
                        melody = predict_gru(
                            gru, input_sequence, sequence_length=64)
                    if msg.note == 107:
                        print("Using Encoder/Decoder LSTM")
                        melody = predict_lstm_enc_dec(
                            lstmencoder, lstmdecoder, input_sequence, sequence_length=64)
                    if msg.note == 108:
                        print("Using Encoder/Decoder GRU")
                        melody = predict_gru_enc_dec(
                            gruencoder, grudecoder, input_sequence, sequence_length=64)

                    print(melody)

                    # Play melody
                    previous_note = None

                    # Sustain Pedal
#                   outport.send(mido.Message(type="control_change", control=64, value=0))
#                   outport.send(mido.Message(type="control_change", control=64, value=127))

                    for note in melody:
                        time.sleep(1./16)
                        if previous_note == None:
                            if note != 0:
                                outport.send(mido.Message(
                                    type="note_on", note=note, velocity=75))

                        elif note != previous_note:
                            if previous_note != 0:
                                outport.send(mido.Message(
                                    type="note_off", note=previous_note))

                            if note != 0:
                                outport.send(mido.Message(
                                    type="note_on", note=note, velocity=75))

                        previous_note = note

                    outport.send(mido.Message(
                        type="note_off", note=previous_note))

                    # Clear buffer
                    print("Clearing note buffer")
                    instrument.notes = []

                else:
                    note_end = wallclock
                    note = pretty_midi.Note(
                        start=note_start, end=note_end, pitch=msg.note, velocity=75)
                    instrument.notes.append(note)
