# from flask import Flask, render_template, Response
# import cv2
# import numpy as np

# app = Flask(__name__)
# cam = cv2.VideoCapture(0)

# # main OMR ratio
# OMR_RATIO = 26/21

# # answer sheet area (adjust later if needed)
# # assuming answers are lower 70% of OMR
# ANSWER_TOP = 0.30

# last_crop = None

# def gen_frames():
#     global last_crop

#     while True:
#         ret, frame = cam.read()
#         if not ret:
#             break

#         h, w, _ = frame.shape

#         box_w = int(w * 0.5)
#         box_h = int(box_w * OMR_RATIO)

#         cx, cy = w//2, h//2

#         x1 = cx - box_w//2
#         y1 = cy - box_h//2
#         x2 = cx + box_w//2
#         y2 = cy + box_h//2

#         crop = frame[y1:y2, x1:x2]
#         last_crop = crop.copy()

#         gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#         thresh = cv2.threshold(gray,0,255,
#                     cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

#         preview = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
#         frame[y1:y2, x1:x2] = preview

#         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
#         cv2.putText(frame,"Place OMR Here",(x1,y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield(b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route('/video')
# def video():
#     return Response(gen_frames(),
#         mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/capture')
# def capture():
#     global last_crop

#     omr = last_crop

#     cv2.imwrite("full_omr.jpg", omr)

#     h, w, _ = omr.shape

#     # answer sheet crop (bottom part)
#     y = int(h * ANSWER_TOP)
#     answer_sheet = omr[y:h, 0:w]

#     cv2.imwrite("answer_sheet.jpg", answer_sheet)

#     cv2.imshow("Captured OMR", omr)
#     cv2.imshow("Final Answer Sheet", answer_sheet)
#     cv2.waitKey(0)

#     return "Captured. Answer sheet saved as answer_sheet.jpg"

# if __name__ == "__main__":
#     app.run(debug=True)










# from flask import Flask, render_template, request, send_file
# import cv2, numpy as np
# from io import BytesIO

# app = Flask(__name__)

# OMR_RATIO = 26/21
# ANSWER_TOP = 0.30   # lower 70% = answer sheet

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/process", methods=["POST"])
# def process():

#     file = request.files["image"].read()
#     img = cv2.imdecode(np.frombuffer(file,np.uint8),1)

#     h,w,_ = img.shape

#     box_w = int(w*0.7)
#     box_h = int(box_w*OMR_RATIO)

#     cx,cy=w//2,h//2

#     x1=cx-box_w//2
#     y1=cy-box_h//2
#     x2=cx+box_w//2
#     y2=cy+box_h//2

#     omr = img[y1:y2,x1:x2]

#     gray=cv2.cvtColor(omr,cv2.COLOR_BGR2GRAY)
#     thresh=cv2.threshold(gray,0,255,
#         cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

#     hh,ww=thresh.shape
#     y=int(hh*ANSWER_TOP)

#     answer = thresh[y:hh,0:ww]

#     answer=cv2.cvtColor(answer,cv2.COLOR_GRAY2BGR)

#     ok,buf=cv2.imencode(".jpg",answer)

#     return send_file(BytesIO(buf),
#         mimetype="image/jpeg")

# if __name__=="__main__":
#     app.run(host="0.0.0.0",port=10000)


# from flask import Flask, render_template, request, send_file
# import cv2, numpy as np
# from io import BytesIO

# app = Flask(__name__)

# OMR_RATIO = 26/21
# ANSWER_TOP = 0.30   # bottom 70% = answer sheet

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/process", methods=["POST"])
# def process():

#     file = request.files["image"].read()
#     img = cv2.imdecode(np.frombuffer(file,np.uint8),1)

#     h,w,_ = img.shape

#     # -------- OMR bounding box (26:21) --------
#     box_w = int(w*0.7)
#     box_h = int(box_w*OMR_RATIO)

#     cx,cy = w//2, h//2

#     x1 = cx-box_w//2
#     y1 = cy-box_h//2
#     x2 = cx+box_w//2
#     y2 = cy+box_h//2

#     omr = img[y1:y2, x1:x2]

#     # -------- Answer sheet ratio crop --------
#     hh,ww,_ = omr.shape
#     y = int(hh * ANSWER_TOP)

#     answer_sheet = omr[y:hh, 0:ww]

#     # encode and return only answer sheet
#     ok, buf = cv2.imencode(".jpg", answer_sheet)

#     return send_file(BytesIO(buf), mimetype="image/jpeg")

# if __name__=="__main__":
#     app.run(host="0.0.0.0", port=10000)
from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

OMR_RATIO = 26 / 21

X_RATIO = 0.005
Y_RATIO = 0.57
W_RATIO = 0.99
H_RATIO = 0.22


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():

    if "image" not in request.files:
        return "no image", 400

    file = request.files["image"].read()
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return "decode failed", 400

    h, w, _ = img.shape

    # --- OMR box ---
    box_w = int(w * 0.7)
    box_h = int(box_w * OMR_RATIO)

    cx, cy = w // 2, h // 2

    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(w, cx + box_w // 2)
    y2 = min(h, cy + box_h // 2)

    omr = img[y1:y2, x1:x2]

    oh, ow, _ = omr.shape

    # --- Answer block ---
    ax = int(ow * X_RATIO)
    ay = int(oh * Y_RATIO)
    aw = int(ow * W_RATIO)
    ah = int(oh * H_RATIO)

    ax = max(0, ax)
    ay = max(0, ay)
    aw = min(ow - ax, aw)
    ah = min(oh - ay, ah)

    answer = omr[ay:ay+ah, ax:ax+aw]

    ok, buf = cv2.imencode(".jpg", answer)
    if not ok:
        return "encode failed", 400

    return send_file(
        BytesIO(buf),
        mimetype="image/jpeg",
        as_attachment=False
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)





