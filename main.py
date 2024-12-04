import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load the DNN model for face detection
MODEL_PATH = "models/"
PROTOTXT = MODEL_PATH + "deploy.prototxt"
CAFFEMODEL = MODEL_PATH + "res10_300x300_ssd_iter_140000_fp16.caffemodel"

face_net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # try:
    while True:
            try:
                payload = await websocket.receive_text()
                payload = json.loads(payload)

                if 'data' not in payload or 'image' not in payload['data']:
                    logging.warning("Invalid input received.")
                    continue

                image_base64 = payload['data']['image'].split(',')[1]

                try:
                    # Decode and convert into image
                    image_data = base64.b64decode(image_base64)
                    np_image = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
                except Exception as e:
                    logging.error(f"Error decoding image: {e}")
                    continue

                if frame is None:
                    logging.error("Failed to decode image.")
                    continue

                try:
                    # Detect faces using DNN model
                    (h, w) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                    face_net.setInput(blob)
                    detections = face_net.forward()

                    for i in range(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]

                        # Confidence threshold
                        if confidence > 0.6:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            # Ensure box is within frame bounds
                            startX = max(0, startX)
                            startY = max(0, startY)
                            endX = min(w, endX)
                            endY = min(h, endY)

                            # Blur the detected face
                            face = frame[startY:endY, startX:endX]
                            if face.size > 0:
                                face = cv2.GaussianBlur(face, (99, 99), 30)
                                frame[startY:endY, startX:endX] = face

                    # Encode the processed frame back to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    blurred_frame_base64 = base64.b64encode(buffer).decode('utf-8')

                    # Send the blurred frame to the frontend
                    response = {
                        "blurred_frame": blurred_frame_base64
                    }
                    await websocket.send_json(response)

                except Exception as e:
                    logging.error(f"Error processing frame: {e}")

            except WebSocketDisconnect:
                logging.info("WebSocket connection closed by the client.")
                break  # Exit the loop if the client disconnects

            except Exception as e:
                logging.error(f"Error receiving message: {e}")
                break  # Exit the loop if any other error occurs

    # finally:
    #     try:
    #         await websocket.close()  # Close the WebSocket connection properly
    #     except RuntimeError as e:
    #         logging.warning(f"Error closing websocket: {e}")  # Handle errors if trying to close an already closed connection
