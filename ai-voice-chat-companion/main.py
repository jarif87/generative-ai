import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import os
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import pyaudio
import wave

# Load environment variables from .env file
load_dotenv()

# Configure the Google AI SDK with the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up the generative model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Initialize Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Voice & Chat AI Companion", page_icon=":dragon:")
st.title("SpeakSmart Intelligent Voice and Chat Assistant")

# Function to get response from the Gemini model
def get_response(query, chat_history):
    history = [{"parts": [{"text": msg['content']}], "role": "user" if msg['type'] == "human" else "model"} for msg in chat_history]
    chat_session = model.start_chat(history=history)
    try:
        response = chat_session.send_message(query)
        return response.text
    except genai.types.StopCandidateException as e:
        return e.candidate.text

# Function to handle voice input using sounddevice and SpeechRecognition
def handle_voice_input():
    temp_audio_file = "temp_audio.wav"

    try:
        st.write("Speak now...")
        fs = 44100  # Sample rate
        seconds = 5  # Duration of recording

        # Ensure the selected audio device is available and accessible
        devices = sd.query_devices()
        if devices:
            device_id = devices[0]['index']  # Use the first available device (modify as needed)
        else:
            st.error("No audio input devices found.")
            return None

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2, device=device_id)
        sd.wait()  # Wait until recording is finished

        # Save the audio to the temporary file
        write_audio(temp_audio_file, myrecording, fs)

        # Use SpeechRecognition to recognize speech from the temporary file
        r = sr.Recognizer()
        with sr.AudioFile(temp_audio_file) as source:
            audio_data = r.record(source)  # Read the entire audio file
            user_query = r.recognize_google(audio_data)
        return user_query
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand what you said.")
        return None
    except sr.RequestError:
        st.write("Sorry, I'm having trouble accessing the Google API.")
        return None
    except PermissionError:
        st.error("Permission error accessing audio device.")
        return None
    except Exception as e:
        st.error(f"Error handling voice input: {e}")
        return None
    finally:
        # Delete the temporary audio file if it exists
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

# Helper function to write audio to file using wave module
def write_audio(filename, data, fs):
    try:
        # Ensure data is in the correct format
        if data.dtype != np.int16:
            data = (data * np.iinfo(np.int16).max).astype(np.int16)

        # Write NumPy array to WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(2)
        wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(fs)
        wf.writeframes(data.tobytes())
        wf.close()
    except Exception as e:
        st.error(f"Error writing audio: {e}")

# Render conversation history
for message in st.session_state.chat_history:
    if message['type'] == "human":
        with st.chat_message("Human"):
            st.markdown(message['content'])
    else:
        with st.chat_message("AI"):
            st.markdown(message['content'])

# User input - handle both text and voice
input_method = st.selectbox("Select input Text or Voice method", ["Text", "Voice"])

user_query = None
if input_method == "Text":
    user_query = st.text_input("Your Message")
    st.write("")  # Clear any previous "Speak now..." message
elif input_method == "Voice":
    user_query = handle_voice_input()
    st.write("")  # Clear any previous "Speak now..." message

# Display voice icon if voice input is selected
if input_method == "Voice":
    st.write("🎤 Voice Input")

# Process user query and AI response
if user_query:
    st.session_state.chat_history.append({"type": "human", "content": user_query})

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history)
        st.markdown(ai_response)

    st.session_state.chat_history.append({"type": "ai", "content": ai_response})
