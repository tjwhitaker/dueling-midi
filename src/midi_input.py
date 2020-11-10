# import sys
# import time
# import pretty_midi
# import utils
# from rtmidi.midiutil import open_midiinput


# class MidiInputHandler(object):
#     def __init__(self, port):
#         self.port = port
#         self.instrument = pretty_midi.Instrument(2)
#         self.start_time = 0
#         self._wallclock = 0.0

#     def __call__(self, event, data=None):
#         message, deltatime = event
#         self._wallclock += deltatime

#         # Trigger Neural Network prediction when lowest key on 88 is pressed
#         if message[1] == 21:
#             roll = self.instrument.get_piano_roll(fs=16)
#             trimmed = utils.trim_roll(roll)
#             pitches, velocities = utils.split_roll(trimmed)
#             print(pitches[-64:])
#         else:
#             if message[0] == 0x90:
#                 self.start_time = self._wallclock

#             elif message[0] == 0x80:
#                 stop_time = self._wallclock
#                 note = pretty_midi.Note(
#                     start=self.start_time, end=stop_time, pitch=message[1], velocity=100)
#                 self.instrument.notes.append(note)


# if __name__ == "__main__":
#     in_port = 5

#     try:
#         midiin, port_name = open_midiinput(in_port)
#     except (EOFError, KeyboardInterrupt):
#         sys.exit()

#     print("Attaching MIDI input callback handler.")
#     midiin.set_callback(MidiInputHandler(port_name))

#     print("Entering main loop. Press Control-C to exit.")
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print('')
#     finally:
#         print("Exit.")
#         midiin.close_port()
#         del midiin

import mido

with mido.open_input() as input_port:
    for msg in input_port:
        print(msg)
