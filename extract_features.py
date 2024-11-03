import pandas as pd
import numpy as np
import librosa


def extract_features(file_name):
    """Extract MFCC features from an audio file."""
    audio, sample_rate = librosa.load(file_name, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)


def create_dataset(csv_file):
    """Create a dataset from the CSV file and extract features."""
    df = pd.read_csv(csv_file)
    features = []
    labels = []

    for index, row in df.iterrows():
        file_name = row['filename']
        label = row['label']
        mfccs = extract_features(file_name)
        features.append(mfccs)
        labels.append(label)

    return np.array(features), np.array(labels)


if __name__ == "__main__":
    features, labels = create_dataset('data.csv')
    np.save('features.npy', features)
    np.save('labels.npy', labels)
