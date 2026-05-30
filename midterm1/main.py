from base_implementation import train_base_models
from improved_implementation import train_improved_models
from cnn_spectro_implementation import train_spectro_cnn_model
from e2e_inference import record_and_classify, evaluate_recordings_folder
from evaluation import show_confusion_matrix

def train_all_and_compare():
    print("\n" + "="*60)
    print("🚀 RUNNING ALL EXPERIMENTS")
    print("   (Existing models will be loaded automatically)")
    print("="*60)
    
    all_metrics = {}
    all_metrics.update(train_base_models())
    all_metrics.update(train_improved_models())
    all_metrics.update(train_spectro_cnn_model())
    
    print("\n" + "="*65)
    print(f"{'Model Name':<25} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
    print("="*65)
    for model_name, (train_acc, test_acc) in all_metrics.items():
        print(f"{model_name:<25} | {train_acc*100:>13.2f}% | {test_acc*100:>13.2f}%")
    print("="*65)

def main_menu():
    while True:
        print("\n" + "="*45)
        print(" 🎤 Speech Recognition Lab CLI Tool")
        print("="*45)
        print("1. Train Base Models")
        print("2. Train Improved Models")
        print("3. Train Spectrogram CNN Model (Deep Learning)")
        print("4. Train ALL & Compare Results (Table)")
        print("5. Record Voice & Classify (CNN Inference)")
        print("6. Evaluate 'my_recordings' Folder")
        print("7. View Confusion Matrix for a Model")
        print("8. Exit")
        
        choice = input("\nSelect an option (1-8): ")
        
        if choice == '1':
            train_base_models()
        elif choice == '2':
            train_improved_models()
        elif choice == '3':
            train_spectro_cnn_model()
        elif choice == '4':
            train_all_and_compare()
        elif choice == '5':
            record_and_classify()
        elif choice == '6':
            evaluate_recordings_folder("my_recordings")
        elif choice == '7':
            show_confusion_matrix()
        elif choice == '8':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main_menu()