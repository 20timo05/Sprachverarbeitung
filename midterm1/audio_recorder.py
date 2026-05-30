import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

# ==========================================
# GLOBAL CONFIGURATION VARIABLES
# ==========================================
ENABLE_MIC_TEST = True
ENABLE_RECORDING = True

TEST_DURATION = 10               # Seconds to run the mic test
RECORDING_DIR = 'my_recordings'  # Directory to save recordings
BASE_FILENAME = 'my_command'     # Prefix for saved audio files

# ==========================================
# FUNCTIONS
# ==========================================

def audio_callback(indata, frames, time_info, status):
    """Callback function to continuously read audio data and print a volume meter."""
    if status:
        print(status, file=sys.stderr)
        
    # Calculate the peak volume of the current audio chunk
    volume = np.max(np.abs(indata))
    
    # Scale the volume for visual representation
    bar_length = int(volume * 100) 
    bar_length = min(bar_length, 50) # Cap at 50 characters
    
    # Create the visual bar string
    meter = '█' * bar_length + '-' * (50 - bar_length)
    
    # Use '\r' to overwrite the same line
    sys.stdout.write(f"\r🎤 Live Mic Level: [{meter}]")
    sys.stdout.flush()

def select_device():
    """Lists available input devices and prompts the user to select one."""
    print("Available input devices:")
    devices = sd.query_devices()
    valid_input_devices =[]
    
    # Filter and list only devices with input channels
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"[{i}] {dev['name']}")
            valid_input_devices.append(i)
            
    if not valid_input_devices:
        print("❌ No input devices found.")
        sys.exit(1)
        
    # Let the user pick a device
    while True:
        try:
            choice = input("\nEnter the ID of the microphone you want to use (or press Enter to use default): ")
            if choice.strip() == "":
                return None  # Tells sounddevice to use the system default
                
            selected_device = int(choice)
            if selected_device in valid_input_devices:
                return selected_device
            else:
                print("Invalid ID. Please select a valid input device ID from the list.")
        except ValueError:
            print("Please enter a valid number.")

def test_microphone(dev_id):
    """Runs a live volume meter for the specified duration to test the mic."""
    dev_str = f"device [{dev_id}]" if dev_id is not None else "the default device"
    print(f"\nStarting mic test on {dev_str} for {TEST_DURATION} seconds. Speak into your microphone!")
    
    try:
        # Open a continuous input stream on the selected device
        with sd.InputStream(device=dev_id, callback=audio_callback, channels=1, samplerate=16000):
            time.sleep(TEST_DURATION)
            
        print("\n\n✅ Mic test finished.")
    except Exception as e:
        print(f"\n\n❌ Error starting stream: {e}")
        print("Note: This might happen if your chosen device doesn't support the requested sample rate (16000 Hz) or channels (1).")

def record_audio(dev_id):
    """Records a 1-second audio clip and saves it to a WAV file."""
    # Ensure the target directory exists
    os.makedirs(RECORDING_DIR, exist_ok=True)
    
    # Find the next available incremented filename
    index = 1
    while True:
        output_filename = f"{BASE_FILENAME}_{index}.wav"
        full_path = os.path.join(RECORDING_DIR, output_filename)
        if not os.path.exists(full_path):
            break
        index += 1

    fs = 16000     # Sample rate
    duration = 1   # Seconds
    channels = 1   # Mono
    
    dev_str = f"device [{dev_id}]" if dev_id is not None else "the default device"
    print(f"\nPlease prepare to say one word (e.g., 'yes', 'no', 'up').")
    print(f"🎙️ Recording will use {dev_str}.")
    print("⏳ Get ready... Recording will start in 3 seconds...")
    sd.sleep(3000) # 3-second wait
    print(f"🔴 RECORDING NOW for {duration} second(s)... Speak!")
    
    try:
        # Record the audio
        my_recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels,
                              dtype='float64', device=dev_id)
        sd.wait() # Wait until recording is complete
        print("✅ Recording complete!")
        
        # Save the recording
        sf.write(full_path, my_recording, fs)
        print(f"📁 Recording saved successfully at: {full_path}")
        
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        print("Note: Ensure the selected device supports the requested sample rate (16000 Hz) and channel count (1).")

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    if not ENABLE_MIC_TEST and not ENABLE_RECORDING:
        print("Both Mic Test and Recording are disabled. Set at least one to True to run.")
        sys.exit(0)

    # 1. Ask for device selection once
    selected_dev_id = select_device()
    
    # 2. Run Mic Test if enabled
    if ENABLE_MIC_TEST:
        test_microphone(selected_dev_id)
    else:
        print("\nMic test is disabled. Set ENABLE_MIC_TEST = True to test your microphone.")
        
    # Optional: Add a pause if both are enabled so the user can prepare for the recording
    if ENABLE_MIC_TEST and ENABLE_RECORDING:
        input("\nPress Enter when you are ready to proceed to the recording phase...")
        
    # 3. Run Recording if enabled
    if ENABLE_RECORDING:
        record_audio(selected_dev_id)
    else:
        print("\nRecording is disabled. Set ENABLE_RECORDING = True to record a new command.")