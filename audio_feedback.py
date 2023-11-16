import numpy as np
import sounddevice as sd

C_NATURAL = 261.63

def generate_beep(frequency, duration, volume=0.5):
    # Calculate the time axis
    t = np.linspace(0, duration, int(duration * 44100), False)
    
    # Generate a sine wave for the given frequency
    beep = volume * np.sin(2 * np.pi * frequency * t)
    
    return beep

def play_beep(beep):
    sd.play(beep)
    sd.wait()

def play_baseline():
    frequency = C_NATURAL
    duration = 1.5
    volume = 0.2
    play(frequency, duration, volume)

def play(freq=C_NATURAL, dur=1, vol=0.1) -> None:
    beep = generate_beep(freq, dur, vol)
    play_beep(beep)

def calibrated_sound():
    play(4 * C_NATURAL, 0.5, 0.1)

def main():
    play(4 * C_NATURAL, 0.5, 0.1)
    # play_baseline()
    frequency = 261.63  # Adjust the frequency in Hz
    duration = 2.2      # Adjust the duration in seconds
    volume = 0.15       # Adjust the volume (0.0 to 1.0)


    # for i in range(100, 200, 10):
    #     beep = generate_beep(i, duration, volume)
    #     play_beep(beep)

if __name__ == "__main__":
    main()
