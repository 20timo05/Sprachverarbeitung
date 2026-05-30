import os
import librosa
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import shared configuration from your helpers file
from helpers import DATA_DIR, ALLOWED_WORDS, MODELS_DIR

def extract_mel_spectrogram(audio_path, sr=16000, n_mels=40, n_fft=1024, hop_length=512):
    """
    Extracts a Log-Mel Spectrogram from an audio file.
    Returns a 2D numpy array of shape (n_mels, number_of_frames).
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Calculate Mel Spectrogram
    S = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    
    # Convert to log scale (decibels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def prepare_spectro_data(data_directory, max_frames=32):
    X, y = [],[]
    files_to_process =[
        os.path.join(r, f) 
        for r, _, fs in os.walk(data_directory) 
        for f in fs if f.endswith('.wav') and os.path.basename(r) in ALLOWED_WORDS
    ]
                    
    for file_path in tqdm(files_to_process, desc="Spectrogram Data Prep"):
        try:
            spectro = extract_mel_spectrogram(file_path)
            
            # Pad or truncate to ensure consistent shape (40, 32)
            if spectro.shape[1] < max_frames:
                pad_width = max_frames - spectro.shape[1]
                spectro = np.pad(spectro, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                spectro = spectro[:, :max_frames]
                
            X.append(spectro)
            y.append(os.path.basename(os.path.dirname(file_path)))
        except Exception:
            pass
            
    return np.array(X), np.array(y)

def train_spectro_cnn_model():
    print("\n--- Preparing Spectrogram CNN Data ---")
    X_raw, y_raw = prepare_spectro_data(DATA_DIR)
    
    # Add channel dimension for CNN input: shape becomes (samples, 40, 32, 1)
    X_reshaped = X_raw[..., np.newaxis]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    # Ensure models directory exists and save LabelEncoder
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    
    num_classes = len(le.classes_)
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    metrics = {}
    model_path = os.path.join(MODELS_DIR, 'cnn_spectro_model.keras')
    
    print("\nTraining/Loading Spectrogram CNN...")
    if os.path.exists(model_path):
        cnn_model = tf.keras.models.load_model(model_path)
        print("✅ Loaded existing Spectrogram CNN from disk.")
    else:
        # Smarter, Deeper CNN Architecture
        cnn_model = models.Sequential([
            layers.BatchNormalization(input_shape=(40, 32, 1)),
            
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2), # Lower dropout early on
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Block 3 (NEW DEEPER LAYER)
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4), # Higher dropout deeper in the network
            
            # Smart Pooling instead of Flatten()
            layers.GlobalAveragePooling2D(),
            
            # Dense Head
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5), # Heavy dropout before the final prediction
            layers.Dense(num_classes, activation='softmax')
        ])
        
        cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # --- CALLBACKS ---
        # 1. Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=8,               # Wait a bit longer so the LR scheduler has time to work
            restore_best_weights=True, 
            verbose=1
        )
        
        # 2. Learning Rate Scheduler
        # If val_loss doesn't improve for 3 epochs, cut the learning rate in half
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-5, 
            verbose=1
        )
        
        # Train the model with both callbacks
        cnn_model.fit(
            X_train, y_train, 
            epochs=100, 
            batch_size=64, 
            validation_split=0.2, 
            callbacks=[early_stopping, lr_scheduler]
        )
        
        # Save the specific Spectrogram model
        cnn_model.save(model_path)
        print(f"✅ Spectrogram CNN saved to {model_path}")
        
    train_loss, train_acc = cnn_model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    metrics['CNN (Spectrogram)'] = (train_acc, test_acc)
    
    print(f"\n======================================")
    print(f" Spectrogram CNN Train Accuracy: {train_acc*100:.2f}%")
    print(f" Spectrogram CNN Test Accuracy : {test_acc*100:.2f}%")
    print(f"======================================")
    
    return metrics

if __name__ == "__main__":
    train_spectro_cnn_model()