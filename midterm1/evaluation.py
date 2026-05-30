import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf

from helpers import DATA_DIR, MODELS_DIR
from base_implementation import prepare_base_data
from improved_implementation import prepare_improved_data
from cnn_spectro_implementation import prepare_spectro_data

def show_confusion_matrix():
    print("\n" + "="*45)
    print(" 📊 Evaluate Model (Confusion Matrix)")
    print("="*45)
    print("1. Base SVC")
    print("2. Base Random Forest")
    print("3. Base KNN")
    print("4. Improved SVC")
    print("5. Improved Random Forest")
    print("6. Improved KNN")
    print("7. Spectrogram CNN")
    print("8. Cancel")
    
    choice = input("\nSelect the model you want to evaluate (1-8): ")
    if choice == '8' or choice not in[str(i) for i in range(1, 8)]:
        return

    le_path = os.path.join(MODELS_DIR, 'label_encoder.joblib')
    if not os.path.exists(le_path):
        print("❌ Error: Label encoder not found. Please train models first.")
        return
        
    le = joblib.load(le_path)

    # ---------------------------------------------------------
    # 1. Base Models
    # ---------------------------------------------------------
    if choice in ['1', '2', '3']:
        X_raw, y_raw = prepare_base_data(DATA_DIR)
        y_encoded = le.transform(y_raw)
        _, X_test, _, y_test = train_test_split(X_raw, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
        
        if choice == '1':
            model_path, title = 'base_svc.joblib', 'Base SVC'
        elif choice == '2':
            model_path, title = 'base_rf.joblib', 'Base Random Forest'
        else:
            model_path, title = 'base_knn.joblib', 'Base KNN'
            
        full_path = os.path.join(MODELS_DIR, model_path)
        if not os.path.exists(full_path):
            print(f"❌ Error: Model not found at {full_path}")
            return
            
        print(f"Generating predictions for {title}...")
        model = joblib.load(full_path)
        y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # 2. Improved Models
    # ---------------------------------------------------------
    elif choice in ['4', '5', '6']:
        X_raw, y_raw = prepare_improved_data(DATA_DIR)
        y_encoded = le.transform(y_raw)
        _, X_test, _, y_test = train_test_split(X_raw, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
        
        if choice == '4':
            model_path, title = 'improved_svc.joblib', 'Improved SVC'
        elif choice == '5':
            model_path, title = 'improved_rf.joblib', 'Improved Random Forest'
        else:
            model_path, title = 'improved_knn.joblib', 'Improved KNN'

        full_path = os.path.join(MODELS_DIR, model_path)
        if not os.path.exists(full_path):
            print(f"❌ Error: Model not found at {full_path}")
            return
            
        model = joblib.load(full_path)
        
        print(f"Generating predictions for {title}...")
        # Note: RF was trained on unscaled data, SVC and KNN on scaled data!
        if choice == '5': 
            y_pred = model.predict(X_test)
        else:
            scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
            scaler = joblib.load(scaler_path)
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            
    # ---------------------------------------------------------
    # 3. Spectrogram CNN
    # ---------------------------------------------------------
    elif choice == '7':
        X_raw, y_raw = prepare_spectro_data(DATA_DIR)
        X_reshaped = X_raw[..., np.newaxis]
        y_encoded = le.transform(y_raw)
        _, X_test, _, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
        
        model_path = os.path.join(MODELS_DIR, 'cnn_spectro_model.keras')
        if not os.path.exists(model_path):
            print(f"❌ Error: Model not found at {model_path}")
            return
            
        print("Generating predictions for Spectrogram CNN...")
        model = tf.keras.models.load_model(model_path)
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        title = "Spectrogram CNN"

    # ---------------------------------------------------------
    # Plotting the Confusion Matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # We use a 'Blues' colormap. The formatting is set to 'd' to show integers.
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='d')
    plt.title(f"Confusion Matrix: {title}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_confusion_matrix()