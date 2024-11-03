from flask import Flask, request, jsonify
import numpy as np
import predict  # Assuming your prediction logic is in this file

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    audio_file = request.files['file']
    audio_path = 'temp.wav'
    audio_file.save(audio_path)
    prediction = predict.predict_sound(audio_path)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
