import os
import librosa
import numpy as np

# Shared Configuration
DATA_DIR = 'speech_commands_v0.02'
MODELS_DIR = 'models'
ALLOWED_WORDS =['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def extract_mfccs(audio_path, sr=16000, n_mfcc=13, n_fft=4096, hop_length=512):
    """
    Extracts MFCCs from an audio file.
    Returns a 2D numpy array of shape (n_mfcc, number_of_frames).
    """
    # sr=None ensures we use the target sampling rate defined in kwargs
    audio, _ = librosa.load(audio_path, sr=sr)
    
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=n_mfcc, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    return mfccs