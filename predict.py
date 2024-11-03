import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the trained model
model = load_model('pet_sound_detection_model.h5')

# Load the trained label encoder
label_encoder = joblib.load('label_encoder.pkl')  # Ensure this file exists and contains your fitted label encoder

def extract_features(file_name):
    """Extract MFCC features from the audio file."""
    audio, sample_rate = librosa.load(file_name, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def predict_sound(file_name):
    """Predict the sound from an audio file."""
    features = extract_features(file_name)
    features = features.reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)  # Get index of the highest predicted value
    return label_encoder.inverse_transform([predicted_index])  # Decode the predicted index to label

# Test the function
if __name__ == "__main__":
    test_file = 'audio_files/cat.wav'  # Update with the path to your test audio file
    print(f"Prediction for {test_file}: {predict_sound(test_file)}")
