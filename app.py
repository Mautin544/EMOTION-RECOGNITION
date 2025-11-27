import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"]

    # Remove "data:image/jpeg;base64,"
    image_data = image_data.split(",")[1]

    # Decode base64 → OpenCV image
    decoded = base64.b64decode(image_data)
    np_data = np.frombuffer(decoded, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # TODO: insert your model inference here
    # emotion = model.predict(processed_img)

    # TEMPORARY fixed output
    emotion = "happy"

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
