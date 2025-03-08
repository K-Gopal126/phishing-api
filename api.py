from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow Chrome Extension to Access API

# Load trained model
model = joblib.load("phishing_detection_model.pkl")

@app.route('/predict', methods=['POST'])  # ✅ Ensure POST Method is Allowed
def predict():
    try:
        data = request.get_json()  # Get JSON Input
        features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy Array
        prediction = model.predict(features)  # Model Prediction
        return jsonify({"prediction": int(prediction[0])})  # Return JSON Response
    except Exception as e:
        return jsonify({"error": str(e)})  # Return Error if Any

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # ✅ Debug Mode for Errors
