# code_01_check_env.py

import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import speech_recognition as sr
import pyttsx3

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Streamlit version: {st.__version__}")
print(f"SpeechRecognition version: {sr.__version__}")
# AttributeError: module 'pyttsx3' has no attribute '__version__'
# print(f"pyttsx3 version: {pyttsx3.__version__}")

# Test pyttsx3 (simple speak)
try:
    engine = pyttsx3.init()
    engine.say("Environment setup complete. Ready for AI assistance systems!")
    # Text-to-Speech test failed: This means you probably do not have eSpeak or eSpeak-ng installed!. PyAudio might be missing or misconfigured.
    engine.runAndWait()
    print("Text-to-Speech test successful.")
except Exception as e:
    print(f"Text-to-Speech test failed: {e}. PyAudio might be missing or misconfigured.")

# A very basic Streamlit check (requires running the app separately)
# This part is just to show Streamlit is imported
# A full Streamlit app would be `streamlit run your_app.py`
print("Streamlit imported successfully. You can try 'streamlit run check_env.py' after adding 'st.write(\"Hello Streamlit!\")' in a function.")