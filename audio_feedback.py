import numpy as np
import sounddevice as sd
import math

C_NATURAL = 261.63

def generate_beep(frequency, duration, volume=0.5) -> None:
    """
    <frequency>:    frequency in Hz
    <duration>:     duration in seconds
    <volume>:       volume from 0.0 to 1.0
    """
    # Calculate the time axis
    t = np.linspace(0, duration, int(duration * 44100), False)
    
    # Generate a sine wave for the given frequency (beep)
    return volume * np.sin(2 * np.pi * frequency * t)

def play_beep(beep) -> None:
    sd.play(beep)
    sd.wait()

def play(freq=C_NATURAL, dur=1, vol=0.1) -> None:
    play_beep(generate_beep(freq, dur, vol))

def play_baseline():
    frequency = C_NATURAL
    duration = 1
    volume = 0.2
    play(frequency, duration, volume)

def calibrated_sound() -> None:
    note = 4 * C_NATURAL
    play(note, 0.1, 0.1)
    play(note, 0.1, 0.1)

    # Wait 1 second
    play(0, 1, 0)

def dist_to_note(distance):
    return (3 * math.exp(-((0.007 * distance) ** 2)) + 1) * C_NATURAL

def play_distance(distance) -> None:
    # print(3 * math.exp(-((0.007 * distance) ** 2)) + 1)
    # print(dist_to_note(distance))
    play(dist_to_note(distance), 0.2, 0.1)
    # play((3 * math.exp(-((0.007 * distance) ** 2)) + 1) * C_NATURAL, 0.1, 0.1)

def main() -> None:
    # play(C_NATURAL, dur=0.1, vol=0.1)
    # play(1.05 * C_NATURAL, 0.1, 0.1)

    # calibrated_sound()
    # play(4 * C_NATURAL, 0.5, 0.1)
    play_baseline()

if __name__ == "__main__":
    main()
