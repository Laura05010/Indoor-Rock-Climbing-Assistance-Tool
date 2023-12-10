import speech_recognition as sr

import audio_feedback

# Create a recognizer instance
recognizer = sr.Recognizer()

# Use the default microphone as the source for audio input
def input_audio_google_api():
    with sr.Microphone() as source:
        print("Speak something...")
        # Adjust for ambient noise if necessary
        recognizer.adjust_for_ambient_noise(source)
        
        # Listen for speech input from the microphone
        audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            # Use Google Web Speech API to convert audio to text
            text = recognizer.recognize_google(audio, language='en-US') # Change language if needed
            print(f"You said: {text}")

            hand_foot, right_left = None, None
            if "hand" in text or "arm" in text:
                hand_foot = 0
            elif "foot" in text or "leg" in text:
                hand_foot = 1
            
            if "right" in text:
                right_left = 0
            elif "left" in text:
                right_left = 1
            
            if (hand_foot is not None) and (right_left is not None):
                return hand_foot, right_left
            else:
                # Alert climber for restatement
                audio_feedback.unknown_audio_input()
                
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
            pass
        except sr.RequestError as e:
            print(f"Error fetching results; {e}")
            pass
        return -1, -1

from transformers import pipeline
import sounddevice as sd
import numpy as np

# Initialize the pipeline for speech recognition
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h")

def record_audio(duration=5, sr=16000):
    # Function to record audio from the microphone
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def process_speech(text):
    hand_foot, right_left = None, None
    if "hand" in text or "arm" in text:
        hand_foot = 0
    elif "foot" in text or "leg" in text:
        hand_foot = 1
        
    if "right" in text:
        right_left = 0
    elif "left" in text:
        right_left = 1
        
    if hand_foot is not None and right_left is not None:
        print(hand_foot, right_left)
        return hand_foot, right_left
    else:
        # Alert for restatement
        print("Unknown audio input.")
        # Perform necessary action for unknown input
        return -1, -1

def input_audio_huggingface(duration=3):
    with sr.Microphone() as source:
        print("Speak something...")
        # Adjust for ambient noise if necessary
        recognizer.adjust_for_ambient_noise(source)
        
        # Listen for speech input from the microphone
        audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            # Use Google Web Speech API to convert audio to text
            text = speech_recognizer(audio)
            transcription = text['text'].lower()
            print(f"You said: {transcription}")

            hand_foot, right_left = None, None
            if "hand" in text or "arm" in text:
                hand_foot = 0
            elif "foot" in text or "leg" in text:
                hand_foot = 1
            
            if "right" in text:
                right_left = 0
            elif "left" in text:
                right_left = 1
            
            if (hand_foot is not None) and (right_left is not None):
                return hand_foot, right_left
            else:
                # Alert climber for restatement
                audio_feedback.unknown_audio_input()
                
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
            pass
        except sr.RequestError as e:
            print(f"Error fetching results; {e}")
            pass
        return -1, -1


    # try:
    #     # # Record audio from the microphone for a specified duration (in seconds)
    #     # audio_data = record_audio(duration)
        
    #     # Perform speech recognition
    #     text = speech_recognizer(audio_data)
    #     transcription = text['text'].lower()
    #     print(f"You said: {transcription}")

    #     # Process the recognized text
    #     return process_speech(transcription)

    # except Exception as e:
    #     print(f"Error: {e}")
    #     pass
    # return -1, -1


def input_audio():
    # return input_audio_google_api()
    return input_audio_huggingface()

def main():
    while True:
        input_audio()

if "__main__" == __name__:
    main()
