import json
import logging
import random
import sys
from datetime import datetime
from typing import Iterator
import cv2
import numpy as np
from keras.models import load_model
from collections import deque
from flask import Flask, Response, render_template, request, stream_with_context


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

random.seed()  # Initialize the random number generator

SEQUENCE_LENGTH=20
LRCN_model=load_model('Models/Model.h5')
CLASSES_LIST = ["Normal","Abnormal"]
video_reader = cv2.VideoCapture(0)
frames_queue = deque(maxlen = SEQUENCE_LENGTH)
LABELS=[0,1]

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/graph")
def graph() -> str:
    return render_template("graph.html")

@app.route("/")
def index() -> str:
    return render_template("cameraview.html")



def generate_frames():
    while True:
        success, frame = video_reader.read()
        framex = cv2.resize(frame, (64, 64))
        normalized_frame = framex / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            now = datetime.now()
            time_now = now.strftime("%H:%M:%S")
            #[[1,2],[2,1]]
            #[[[1,2],[2,1]]]
            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            print(predicted_labels_probabilities)
            #[0.56 0.45]
            #[1 0]

            prediction=np.argmax(predicted_labels_probabilities)
            print(CLASSES_LIST[prediction])
            frames_queue.clear()


        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_random_data() -> Iterator[str]:
    """
    Generates random value between 0 and 100

    :return: String containing current timestamp (YYYY-mm-dd HH:MM:SS) and randomly generated data.
    """
    if request.headers.getlist("X-Forwarded-For"):
        client_ip = request.headers.getlist("X-Forwarded-For")[0]
    else:
        client_ip = request.remote_addr or ""

    try:
        logger.info("Client %s connected", client_ip)
        while True:
            ok, frame = video_reader.read()
            frame = cv2.resize(frame, (64, 64))
            normalized_frame = frame / 255
            frames_queue.append(normalized_frame)
            # Check if the number of frames in the queue are equal to the fixed sequence length.
            if len(frames_queue) == SEQUENCE_LENGTH:
                # Pass the normalized frames to the model and get the predicted probabilities.
                predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]

                # Get the index of class with highest probability.
                predicted_label = np.argmax(predicted_labels_probabilities)
                print(CLASSES_LIST[predicted_label])

                json_data = json.dumps(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "value": int(predicted_label),
                    }
                )
                yield f"data:{json_data}\n\n"

    except GeneratorExit:
        logger.info("Client %s disconnected", client_ip)


@app.route("/chart-data")
def chart_data() -> Response:
    response = Response(stream_with_context(generate_random_data()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


if __name__ == "__main__":
    app.run(threaded=True)
