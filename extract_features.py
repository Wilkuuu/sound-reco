import pandas as pd
import librosa
import numpy as np

# Load data
data_file = 'dat_with_labels.csv'  # This is the updated CSV file
df = pd.read_csv(data_file).head(100)

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Function to create dataset
def create_dataset():
    features = []
    labels = []
    for index, row in df.iterrows():
        # Assuming you have a way to fetch or generate audio files using YTID and start/end seconds
        audio_file = f"/media/wilk/goodram/archive/archive/train_wav/{row['YTID']}.wav"  # Example path
        try:
            mfccs = extract_features(audio_file)
            features.append(mfccs)
            labels.append(row['positive_labels_mapped'])  # Use mapped labels here
        except FileNotFoundError:
            print(f"File not found: {audio_file}")
            continue

    return np.array(features), labels

# Run the dataset creation
features, labels = create_dataset()
np.save('features.npy', features)
np.save('labels.npy', labels)
print("Features and labels saved.")
