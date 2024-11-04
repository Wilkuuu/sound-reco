import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

from extract_features import extract_features

# Load model
model = tf.keras.models.load_model('pet_sound_detection_model.keras')
print(model.summary())

# Load labels and ensure all unique classes are used in the encoder
labels = np.load('labels.npy', allow_pickle=True)
unique_labels = set(label for sublist in labels for label in sublist.split(', '))
mlb = MultiLabelBinarizer(classes=list(unique_labels))
mlb.fit([label.split(', ') for label in labels])  # Fit on all labels


# Predict function
def predict_sound(file_name):
    # Extract features as done in extract_features.py
    features = extract_features(file_name).reshape(1, -1)  # Reshape for single sample

    # Predict and threshold labels
    prediction = model.predict(features)

    # Check prediction shape matches number of classes
    if prediction.shape[1] != len(mlb.classes_):
        raise ValueError(
            f"Prediction output shape {prediction.shape} does not match number of classes: {len(mlb.classes_)}")

    # Convert prediction to binary indicators and inverse transform
    predicted_labels = mlb.inverse_transform(prediction > 0.5)  # Threshold at 0.5
    return predicted_labels


# Example usage
print(predict_sound("audio_files/example_audio.wav"))
