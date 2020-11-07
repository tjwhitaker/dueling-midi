"""Show how to receive MIDI input by setting a callback function."""

import logging
import sys
import time
import pretty_midi
import collections
import utils

from rtmidi.midiutil import open_midiinput

log = logging.getLogger('midiin_callback')
logging.basicConfig(level=logging.DEBUG)


class MidiInputHandler(object):
    def __init__(self, port):
        self.port = port
        self.instrument = pretty_midi.Instrument(2)
        self.start_time = 0
        self._wallclock = 0.0
        self.count = 0

    def __call__(self, event, data=None):
        message, deltatime = event
        self._wallclock += deltatime


        # IMPORTANT: Only works monophonically
        if message[0] == 0x90:
            self.start_time = self._wallclock
        
        elif message[0] == 0x80:
            self.count += 1
            stop_time = self._wallclock
            note = pretty_midi.Note(start=self.start_time, end=stop_time, pitch=message[1], velocity=100)
            self.instrument.notes.append(note)

        
        if self.count > 32:
            roll = self.instrument.get_piano_roll()
            trimmed = utils.trim_roll(roll)
            pitches, velocities = utils.split_roll(trimmed)
            print(pitches[-64:])

        


# Prompts user for MIDI input port, unless a valid port number or name
# is given as the first argument on the command line.
# API backend defaults to ALSA on Linux.
port = 5

try:
    midiin, port_name = open_midiinput(port)
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
