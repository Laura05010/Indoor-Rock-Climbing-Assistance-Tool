import speech_recognition as sr

# Create a recognizer instance
recognizer = sr.Recognizer()

# Use the default microphone as the source for audio input
def input_audio():
    with sr.Microphone() as source:
        while True:
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
            except sr.UnknownValueError:
                print("Sorry, could not understand the audio.")
            except sr.RequestError as e:
                print(f"Error fetching results; {e}")

def main():
    input_audio()

if "__main__" == __name__:
    main()