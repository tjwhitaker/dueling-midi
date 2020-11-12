from predict_lstm import predict_lstm
from models import NoteLSTM
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
note_start=0

# Pretty midi for piano roll
# Probably a good place to improve performance.
# TODO: Better way to generate sequence of notes.
instrument = pretty_midi.Instrument(2)

print("Setting up the model")

# Setting up the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NoteLSTM().to(device)
model.load_state_dict(torch.load("../models/notelstm.model"))
model.eval()

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
                if msg.note == 21:
                    # Generate Melody
                    print("Generating melody")
                    roll = instrument.get_piano_roll(fs=32)
                    trimmed = utils.trim_roll(roll)
                    pitches, _ = utils.split_roll(trimmed)

                    input_sequence = torch.tensor([pitches[-64:]]).to(device)
                    melody = predict_lstm(model, input_sequence)

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
                        time.sleep(1./32)

                    outport.send(mido.Message(
                        type="note_off", note=previous_note))

                else:
                    note_end = wallclock
                    note = pretty_midi.Note(
                        start=note_start, end=note_end, pitch=msg.note, velocity=100)
                    instrument.notes.append(note)
