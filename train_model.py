import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import joblib

# Load features and labels
features = np.load('features.npy')
labels = np.load('labels.npy', allow_pickle=True)

# Convert labels to a list of lists for multi-label binarization
labels = [label.split(', ') for label in labels]  # Split string labels into list of labels

# Encode labels with MultiLabelBinarizer
mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(labels)

# Save the MultiLabelBinarizer for future use
joblib.dump(mlb, 'multi_label_binarizer.pkl')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(mlb.classes_), activation='sigmoid'))  # Use sigmoid for multi-label classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Use binary crossentropy for multi-label

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('pet_sound_detection_model.keras')
