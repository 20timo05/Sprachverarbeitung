import os
import time
import numpy as np
import joblib
import sounddevice as sd
import soundfile as sf
import tensorflow as tf

# Import MODELS_DIR from helpers, and the new spectrogram extractor from the CNN script
from helpers import MODELS_DIR
from cnn_spectro_implementation import extract_mel_spectrogram

def record_and_classify():
    print("\n--- End-to-End CNN Voice Classification ---")
    model_path = os.path.join(MODELS_DIR, 'cnn_spectro_model.keras')
    le_path = os.path.join(MODELS_DIR, 'label_encoder.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        print("❌ Error: Spectrogram CNN Model or LabelEncoder not found. Please train the CNN model first!")
        return

    fs = 16000
    duration = 1.0
    print("\n🎙️ Get ready to say a command (e.g., 'yes', 'no', 'up', 'down')...")
    time.sleep(2)
    print("🔴 RECORDING NOW (1 second) - Speak!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    print("✅ Recording complete.")
    
    temp_wav = "live_command.wav"
    sf.write(temp_wav, recording, fs)
    
    # Preprocess (Using Spectrograms now!)
    spectro = extract_mel_spectrogram(temp_wav, sr=fs)
    max_frames = 32
    if spectro.shape[1] < max_frames:
        pad_width = max_frames - spectro.shape[1]
        spectro = np.pad(spectro, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        spectro = spectro[:, :max_frames]
        
    # New Input Shape: 40 Mel bands instead of 13 MFCCs
    cnn_input = spectro.reshape(1, 40, 32, 1)
    
    try:
        model = tf.keras.models.load_model(model_path)
        le = joblib.load(le_path)
        
        pred_probs = model.predict(cnn_input, verbose=0)
        pred_idx = np.argmax(pred_probs, axis=1)[0]
        confidence = pred_probs[0][pred_idx] * 100
        pred_word = le.inverse_transform([pred_idx])[0]
        
        print(f"\n=============================================")
        print(f"🤖 CNN Predicts: {pred_word.upper()} (Confidence: {confidence:.2f}%)")
        print(f"=============================================\n")
    except Exception as e:
        print(f"Prediction failed: {e}")

def evaluate_recordings_folder(folder_path="my_recordings"):
    print(f"\n--- Batch Evaluating Folder: {folder_path} ---")
    model_path = os.path.join(MODELS_DIR, 'cnn_spectro_model.keras')
    le_path = os.path.join(MODELS_DIR, 'label_encoder.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        print("❌ Error: Spectrogram CNN Model or LabelEncoder not found. Please train the CNN model first!")
        return
        
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' does not exist.")
        return
        
    wav_files =[f for f in os.listdir(folder_path) if f.endswith('.wav')]
    if not wav_files:
        print(f"No .wav files found in '{folder_path}'.")
        return
        
    model = tf.keras.models.load_model(model_path)
    le = joblib.load(le_path)
    
    print(f"\n{'Filename':<20} | {'Assumed Target':<15} | {'CNN Prediction':<15} | {'Confidence'}")
    print("-" * 75)
    
    for file in wav_files:
        file_path = os.path.join(folder_path, file)
        
        # Preprocess (Using Spectrograms now!)
        spectro = extract_mel_spectrogram(file_path, sr=16000)
        
        max_frames = 32
        if spectro.shape[1] < max_frames:
            pad_width = max_frames - spectro.shape[1]
            spectro = np.pad(spectro, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            spectro = spectro[:, :max_frames]
            
        # New Input Shape: 40 Mel bands instead of 13 MFCCs
        cnn_input = spectro.reshape(1, 40, 32, 1)
        
        pred_probs = model.predict(cnn_input, verbose=0)
        pred_idx = np.argmax(pred_probs, axis=1)[0]
        confidence = pred_probs[0][pred_idx] * 100
        pred_word = le.inverse_transform([pred_idx])[0]
        
        # Guesses the target word from the filename (e.g. "yes_1.wav" -> "yes")
        actual_guess = file.split('_')[0].split('.')[0].lower() 
        
        # Add visual indicator if the prediction matched the filename
        match_icon = "✅" if actual_guess == pred_word else "❌"
        
        print(f"{file:<20} | {actual_guess:<15} | {pred_word.upper():<15} | {confidence:>6.2f}% {match_icon}")
    print("-" * 75)