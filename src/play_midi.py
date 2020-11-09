import numpy as np
import torch
from models import NoteLSTM
import utils
import rtmidi
import time


def play_midi(melody, midi_out):
    with midiout:
        for note in melody:
            note_on = [0x90, note, 112]
            note_off = [0x80, note, 0]

            if note == 0:
                note_on = [0x90, note, 0]

            midiout.send_message(note_on)
            time.sleep(0.25)
            midiout.send_message(note_off)
