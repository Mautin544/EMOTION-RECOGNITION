from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

model = tf.keras.models.load_model('face_emotionModel.h5') 
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Happy', 'Neutral', 'Sad', 'Surprise']

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_haar_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3, 
                minNeighbors=5,
                minSize=(50, 50)  # helps avoid false detections
            )

            for (x, y, w, h) in faces:
                # Draw blue rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 3)

                # Crop face
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype("float") / 255.0
                roi_gray = img_to_array(roi_gray)           # shape: (48,48,1)
                roi_gray = np.expand_dims(roi_gray, axis=0)  # shape: (1,48,48,1)

                # Predict emotion
                prediction = model.predict(roi_gray, verbose=0)[0]
                max_index = np.argmax(prediction)
                emotion = emotions[max_index]
                confidence = prediction[max_index]

                # Show emotion with bigger, clearer text
                label = f"{emotion} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-15), 
                           cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)

                # Optional: green confidence bar at bottom
                bar_len = int(w * confidence)
                cv2.rectangle(frame, (x, y+h-12), (x + bar_len, y+h), (0, 255, 0), -1)

            # Encode and stream
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)