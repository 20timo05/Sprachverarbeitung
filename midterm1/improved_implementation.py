import os
import librosa
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from helpers import extract_mfccs, DATA_DIR, ALLOWED_WORDS, MODELS_DIR

def prepare_improved_data(data_directory):
    X, y = [],[]
    files_to_process =[os.path.join(r, f) for r, _, fs in os.walk(data_directory) for f in fs if f.endswith('.wav') and os.path.basename(r) in ALLOWED_WORDS]
                    
    for file_path in tqdm(files_to_process, desc="Improved Data Prep"):
        try:
            mfccs = extract_mfccs(file_path)
            delta_mfccs = librosa.feature.delta(mfccs)
            combined_features = np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1), np.mean(delta_mfccs, axis=1)])
            X.append(combined_features)
            y.append(os.path.basename(os.path.dirname(file_path)))
        except Exception:
            pass
    return np.array(X), np.array(y)

def train_improved_models():
    print("\n--- Preparing Improved Data ---")
    X_raw, y_raw = prepare_improved_data(DATA_DIR)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
    
    if os.path.exists(os.path.join(MODELS_DIR, 'scaler.joblib')):
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        X_train_scaled = scaler.transform(X_train)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    X_test_scaled = scaler.transform(X_test)
    
    metrics = {}
    
    print("\nTraining/Loading Improved SVC...")
    if os.path.exists(os.path.join(MODELS_DIR, 'improved_svc.joblib')):
        svm_model = joblib.load(os.path.join(MODELS_DIR, 'improved_svc.joblib'))
    else:
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm_model.fit(X_train_scaled, y_train)
        joblib.dump(svm_model, os.path.join(MODELS_DIR, 'improved_svc.joblib'))
    metrics['Improved SVC'] = (accuracy_score(y_train, svm_model.predict(X_train_scaled)), accuracy_score(y_test, svm_model.predict(X_test_scaled)))

    print("Training/Loading Improved Random Forest...")
    if os.path.exists(os.path.join(MODELS_DIR, 'improved_rf.joblib')):
        rf_model = joblib.load(os.path.join(MODELS_DIR, 'improved_rf.joblib'))
    else:
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, os.path.join(MODELS_DIR, 'improved_rf.joblib'))
    metrics['Improved RF'] = (accuracy_score(y_train, rf_model.predict(X_train)), accuracy_score(y_test, rf_model.predict(X_test)))

    print("Training/Loading Improved KNN...")
    if os.path.exists(os.path.join(MODELS_DIR, 'improved_knn.joblib')):
        knn_model = joblib.load(os.path.join(MODELS_DIR, 'improved_knn.joblib'))
    else:
        knn_model = KNeighborsClassifier(n_neighbors=11)
        knn_model.fit(X_train_scaled, y_train)
        joblib.dump(knn_model, os.path.join(MODELS_DIR, 'improved_knn.joblib'))
    metrics['Improved KNN'] = (accuracy_score(y_train, knn_model.predict(X_train_scaled)), accuracy_score(y_test, knn_model.predict(X_test_scaled)))

    return metrics

if __name__ == "__main__":
    train_improved_models()