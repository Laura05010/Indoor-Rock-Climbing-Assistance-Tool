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
import torchaudio
import torch

# Initialize the pipeline for speech recognition
speech_recognizer = pipeline("automatic-speech-recognition", 
                             model="facebook/wav2vec2-large-960h")

def input_audio_hugging_face():
    """
    Dependencies:
    > pip install transformers datasets torchaudio
    """
    try:
        # Use torchaudio to capture speech from the microphone
        microphone = torchaudio.sox_effects.SoxEffectsChain()
        microphone.set_input_format(rate=16000, channels=1, precision=16)
        microphone.set_output_format(rate=16000, channels=1, precision=16)
        microphone.set_effects([{'input': ('mic',), 'output': ('stdout',)}])
        print("Speak something...")
        mic_audio, _ = microphone.sox_build_flow_effects()
        mic_audio = torch.from_numpy(mic_audio.numpy().astype('float32'))
        
        # Perform speech recognition
        text = speech_recognizer(mic_audio)
        print(f"You said: {text[0]['transcription']}")
        
        # Perform your specific logic based on recognized text here
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
        
    except Exception as e:
        print(f"Error: {e}")
        # Handle the error or return default values
        return -1, -1

def input_audio():
    # input_audio_google_api()
    # input_audio_hugging_face()
    n = 0

def main():
    input_audio()

if "__main__" == __name__:
    main()