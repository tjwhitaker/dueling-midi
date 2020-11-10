import mido
import pretty_midi
import time
import utils
import torch
from models import NoteCNN
from predict_cnn import predict_cnn

# Params for midi streaming
port_name = "USB Midi:USB Midi MIDI 1 28:0"
note_buffer = []
start_time = time.time()
wallclock = 0

# Pretty midi for piano roll
# Probably a good place to improve performance.
# TODO: Better way to generate sequence of notes.
instrument = pretty_midi.Instrument(2)

print("Setting up the model")

# Setting up the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NoteCNN().to(device)
model.load_state_dict(torch.load("../models/notecnn.model"))
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
                    roll = instrument.get_piano_roll(fs=16)
                    trimmed = utils.trim_roll(roll)
                    pitches, _ = utils.split_roll(trimmed)

                    input_sequence = torch.tensor([pitches[-64:]]).to(device)
                    melody = predict_cnn(model, input_sequence)

                    # Play melody
                    for note in melody:
                        outport.send(mido.Message(type="note_on", note=note))
                        time.sleep(1./16)
                        outport.send(mido.Message(type="note_off", note=note))
                        time.sleep(1./16)
                else:
                    note_end = wallclock
                    note = pretty_midi.Note(
                        start=note_start, end=note_end, pitch=msg.note, velocity=100)
                    instrument.notes.append(note)
