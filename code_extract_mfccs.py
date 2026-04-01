import librosa

def extract_mfccs(audio_path, sr=16000, n_mfcc=256, n_fft=4096, hop_length=512):
    """ Extracts MFCCs from an audio file.
    Args:
    audio_path (str): Path to the audio file.
    sr (int): Target sampling rate (default: 16000 Hz for speech commands).
    n_mfcc (int): Number of MFCCs to extract.
    n_fft (int): Size of the FFT window.
    hop_length (int): Step size between consecutive frames.

    Returns:
    numpy.ndarray: Array of MFCCs (shape: n_mfcc, number_of_frames).

    saved file names convention:
    Figure1_yes_sr16000_n_mfcc13_n_fft2048_hop_length512.png
    """

    audio, _ = librosa.load(audio_path, sr=sr) 

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length) 
    return mfccs
