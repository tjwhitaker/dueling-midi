import mido
import time

port_name = "USB Midi:USB Midi MIDI 1 28:0"

with mido.open_output(port_name) as outport:
    with mido.open_input(port_name) as inport:
        for msg in inport:
            print(msg)
            time.sleep(0.5)
            outport.send(msg)
