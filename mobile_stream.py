import cv2
import imutils
import numpy as np
from flask import Flask, Response, render_template_string

app = Flask(__name__)

cap = cv2.VideoCapture(0)

latest_frame = None

HTML = """
<html>
<head>
<title>OMR Camera</title>
</head>
<body>
<h2>Live OMR Camera</h2>
<img src="/video">
<br><br>
<form action="/capture">
<button type="submit">📸 Capture</button>
</form>
</body>
</html>
"""

def gen_frames():
    global latest_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=900)
        latest_frame = frame.copy()

        # ---- LIVE THRESHOLD PREVIEW ----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)

        th = cv2.adaptiveThreshold(
            blur,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,7)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts)>0:
            c = max(cnts,key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            cv2.drawContours(frame,[box],-1,(0,255,0),2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    global latest_frame

    if latest_frame is not None:
        cv2.imwrite("captured.jpg", latest_frame)
        return "Captured! Image saved as captured.jpg"

    return "No frame yet"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
