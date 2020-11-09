"""Show how to receive MIDI input by setting a callback function."""

import logging
import sys
import time
import pretty_midi
import collections
import utils
import torch
from models import NoteLSTM
from predict_lstm import predict_lstm
from play_midi import play_midi

from rtmidi.midiutil import open_midiinput, open_midioutput

log = logging.getLogger('midiin_callback')
logging.basicConfig(level=logging.DEBUG)


class MidiInputHandler(object):
    def __init__(self, port):
        self.port = port
        self.instrument = pretty_midi.Instrument(2)
        self.start_time = 0
        self._wallclock = 0.0
        self.midiout = rtmidi.MidiOut()
        self.midiout.open_port(5)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = NoteLSTM().to(self.device)
        self.model.load_state_dict(torch.load("../models/notelstm.model"))
        self.model.eval()

    def __call__(self, event, data=None):
        message, deltatime = event
        self._wallclock += deltatime

        # Trigger Neural Network prediction when lowest key on 88 is pressed
        if message[1] == 21:
            roll = self.instrument.get_piano_roll(fs=16)
            trimmed = utils.trim_roll(roll)
            pitches, velocities = utils.split_roll(trimmed)
            print(pitches[-64:])

            input_sequence = torch.tensor([pitches[-64:]]).to(self.device)

            melody = predict_lstm(self.model, input_sequence)
            play_midi(melody, self.midiout)
        else:
            if message[0] == 0x90:
                self.start_time = self._wallclock

            elif message[0] == 0x80:
                stop_time = self._wallclock
                note = pretty_midi.Note(
                    start=self.start_time, end=stop_time, pitch=message[1], velocity=100)
                self.instrument.notes.append(note)


# Prompts user for MIDI input port, unless a valid port number or name
# is given as the first argument on the command line.
# API backend defaults to ALSA on Linux.
in_port = 5
out_port = 5

try:
    midiin, port_name = open_midiinput(in_port)
    midiout, port_name = open_midioutput(out_port)

except (EOFError, KeyboardInterrupt):
    sys.exit()

print("Attaching MIDI input callback handler.")
midiin.set_callback(MidiInputHandler(port_name))

print("Entering main loop. Press Control-C to exit.")
try:
    # Just wait for keyboard interrupt,
    # everything else is handled via the input callback.
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('')
finally:
    print("Exit.")
    midiin.close_port()
    del midiin
