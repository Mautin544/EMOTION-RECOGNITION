import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained emotion model
model = load_model('face_emotionModel.h5')

# Face labels for FER-2013 dataset
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image from frontend
        data = request.get_json()
        image_data = data["image"].split(",")[1]

        # Decode base64 to numpy array
        decoded = base64.b64decode(image_data)
        np_data = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        # Safety check
        if frame is None:
            return jsonify({"emotion": "Invalid image frame"})

        # Resize to improve detection on Render
        frame = cv2.resize(frame, (640, 480))

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # 🔥 Boost contrast (important on Render)

        # Improved detection settings for weak webcam images
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,       # LOWER for Render
            minSize=(40, 40)      # Larger min size helps
        )

        print("Faces detected:", len(faces))  # Debug log (see Render logs)

        if len(faces) == 0:
            return jsonify({"emotion": "No face detected"})

        # Use the first detected face
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]

        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = roi_gray.reshape(1, 48, 48, 1)

        # Predict emotion
        prediction = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(prediction)]

        return jsonify({"emotion": emotion})

    except Exception as e:
        print("Error:", e)
        return jsonify({"emotion": "Error processing image"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
