# https://github.com/Uberi/speech_recognition
# % pip install SpeechRecognition
import speech_recognition as sr
from enum import Enum
import audio_feedback

class Language (Enum) :
    ENGLISH = "en-US"
    CHINESE = "zh-TW"
    FRENCH = "fr-FR"
    SPANISH_SPAIN = "es-ES"
    SPANISH_LATAM = "es-US"
    KOREAN = "ko-KR"
    JAPANESE = "ja-JP"

def list_mic_device_index():
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print("{1}, device_index={0}".format(index, name))

def speech_to_text(device_index, language=Language.ENGLISH):
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=device_index) as source:
        print("Start Speaking:")

        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language=language.value)
            print("You said: {}".format(text))

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
                audio_feedback.instruction_confirmed()
                return hand_foot, right_left
            else:
                # Alert climber for restatement
                audio_feedback.no_instruction()
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
            audio_feedback.unknown_audio_input()
            pass
        except sr.RequestError as e:
            print(f"Error fetching results; {e}")
            audio_feedback.unknown_audio_input()
            pass
        except:
            # Alert climber for restatement
            print("Please try again.")
            audio_feedback.unknown_audio_input()
    return -1, -1

def main():
    list_mic_device_index()
    while True:
        print(speech_to_text(device_index=0))

if "__main__" == __name__:
    main()
