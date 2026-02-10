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
#     app.run(host="0.0.0.0", port=10000)from flask import Flask, render_template, request, send_file

# from flask import Flask, render_template, request, send_file
# import base64
# from io import BytesIO

# import cv2
# import numpy as np

# app = Flask(__name__)

# OMR_RATIO = 26.5 / 21.5
# BOX_WIDTH_RATIO = 0.7
# X_RATIO = 0.005
# Y_RATIO = 0.57
# W_RATIO = 0.99
# H_RATIO = 0.22


# def _safe_center_box(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
#     max_width_by_height = int(frame_height / OMR_RATIO)
#     box_w = min(int(frame_width * BOX_WIDTH_RATIO), max_width_by_height)
#     box_h = int(box_w * OMR_RATIO)

#     x1 = max(0, min(frame_width - box_w, (frame_width - box_w) // 2))
#     y1 = max(0, min(frame_height - box_h, (frame_height - box_h) // 2))
#     x2 = x1 + box_w
#     y2 = y1 + box_h

#     return x1, y1, x2, y2


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/process", methods=["POST"])
# def process():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("img") or request.form.get("img")
#     if not data:
#         return "no image", 400

#     _, _, encoded = data.partition(",")
#     if not encoded:
#         return "invalid image data", 400

#     try:
#         img_bytes = base64.b64decode(encoded)
#     except (base64.binascii.Error, ValueError):
#         return "decode failed", 400

#     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
#     if img is None:
#         return "decode failed", 400

#     h, w, _ = img.shape
#     x1, y1, x2, y2 = _safe_center_box(w, h)
#     omr = img[y1:y2, x1:x2]

#     oh, ow, _ = omr.shape
#     ax = int(ow * X_RATIO)
#     ay = int(oh * Y_RATIO)
#     aw = int(ow * W_RATIO)
#     ah = int(oh * H_RATIO)

#     ax = max(0, ax)
#     ay = max(0, ay)
#     aw = max(1, min(ow - ax, aw))
#     ah = max(1, min(oh - ay, ah))

#     answer = omr[ay : ay + ah, ax : ax + aw]

#     ok, buf = cv2.imencode(".jpg", answer)
#     if not ok:
#         return "encode failed", 400

#     return send_file(BytesIO(buf), mimetype="image/jpeg")


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=10000)


# from flask import Flask, render_template, request, send_file
# import base64
# from io import BytesIO

# import cv2
# import numpy as np

# app = Flask(__name__)

# OMR_RATIO = 26.5 / 21.5
# BOX_WIDTH_RATIO = 0.7
# X_RATIO = 0.005
# Y_RATIO = 0.57
# W_RATIO = 0.99
# H_RATIO = 0.22


# def _safe_center_box(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
#     max_width_by_height = int(frame_height / OMR_RATIO)
#     box_w = min(int(frame_width * BOX_WIDTH_RATIO), max_width_by_height)
#     box_h = int(box_w * OMR_RATIO)

#     x1 = max(0, min(frame_width - box_w, (frame_width - box_w) // 2))
#     y1 = max(0, min(frame_height - box_h, (frame_height - box_h) // 2))
#     x2 = x1 + box_w
#     y2 = y1 + box_h

#     return x1, y1, x2, y2


# def _order_points(points: np.ndarray) -> np.ndarray:
#     rect = np.zeros((4, 2), dtype="float32")
#     s = points.sum(axis=1)
#     diff = np.diff(points, axis=1)

#     rect[0] = points[np.argmin(s)]
#     rect[2] = points[np.argmax(s)]
#     rect[1] = points[np.argmin(diff)]
#     rect[3] = points[np.argmax(diff)]

#     return rect


# def _warp_omr_like_scanner(omr_candidate: np.ndarray) -> np.ndarray:
#     gray = cv2.cvtColor(omr_candidate, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 60, 180)

#     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

#     for contour in contours:
#         perimeter = cv2.arcLength(contour, True)
#         if perimeter == 0:
#             continue

#         approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
#         if len(approx) != 4:
#             continue

#         area = cv2.contourArea(approx)
#         if area < 0.25 * omr_candidate.shape[0] * omr_candidate.shape[1]:
#             continue

#         quad = approx.reshape(4, 2).astype("float32")
#         rect = _order_points(quad)
#         (tl, tr, br, bl) = rect

#         width_top = np.linalg.norm(tr - tl)
#         width_bottom = np.linalg.norm(br - bl)
#         max_width = int(max(width_top, width_bottom))

#         height_right = np.linalg.norm(br - tr)
#         height_left = np.linalg.norm(bl - tl)
#         max_height = int(max(height_right, height_left))

#         if max_width < 10 or max_height < 10:
#             continue

#         dst = np.array(
#             [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
#             dtype="float32",
#         )
#         matrix = cv2.getPerspectiveTransform(rect, dst)
#         return cv2.warpPerspective(omr_candidate, matrix, (max_width, max_height))

#     return omr_candidate


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/process", methods=["POST"])
# def process():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("img") or request.form.get("img")
#     if not data:
#         return "no image", 400

#     _, _, encoded = data.partition(",")
#     if not encoded:
#         return "invalid image data", 400

#     try:
#         img_bytes = base64.b64decode(encoded)
#     except (base64.binascii.Error, ValueError):
#         return "decode failed", 400

#     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
#     if img is None:
#         return "decode failed", 400

#     h, w, _ = img.shape
#     x1, y1, x2, y2 = _safe_center_box(w, h)
#     omr = img[y1:y2, x1:x2]

#     scanned = _warp_omr_like_scanner(omr)

#     oh, ow, _ = scanned.shape
#     ax = int(ow * X_RATIO)
#     ay = int(oh * Y_RATIO)
#     aw = int(ow * W_RATIO)
#     ah = int(oh * H_RATIO)

#     ax = max(0, ax)
#     ay = max(0, ay)
#     aw = max(1, min(ow - ax, aw))
#     ah = max(1, min(oh - ay, ah))

#     answer = scanned[ay : ay + ah, ax : ax + aw]

#     ok, buf = cv2.imencode(".jpg", answer)
#     if not ok:
#         return "encode failed", 400

#     return send_file(BytesIO(buf), mimetype="image/jpeg")


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=10000)

from flask import Flask, render_template, request, send_file, jsonify
import base64
from io import BytesIO

import cv2
import numpy as np

app = Flask(__name__)

OMR_RATIO = 26.5 / 21.5
BOX_WIDTH_RATIO = 0.7
X_RATIO = 0.005
Y_RATIO = 0.57
W_RATIO = 0.99
H_RATIO = 0.22


def _safe_center_box(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    max_width_by_height = int(frame_height / OMR_RATIO)
    box_w = min(int(frame_width * BOX_WIDTH_RATIO), max_width_by_height)
    box_h = int(box_w * OMR_RATIO)

    x1 = max(0, min(frame_width - box_w, (frame_width - box_w) // 2))
    y1 = max(0, min(frame_height - box_h, (frame_height - box_h) // 2))
    x2 = x1 + box_w
    y2 = y1 + box_h

    return x1, y1, x2, y2


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def _decode_data_url_image(data: str) -> np.ndarray | None:
    _, _, encoded = data.partition(",")
    if not encoded:
        return None

    try:
        img_bytes = base64.b64decode(encoded)
    except (base64.binascii.Error, ValueError):
        return None

    return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)


def _is_ratio_close(ratio: float, expected: float, tolerance: float = 0.40) -> bool:
    return abs(ratio - expected) <= expected * tolerance


def _find_document_quad(image: np.ndarray, expected_ratio: float | None = None) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    candidates = []

    edges = cv2.Canny(enhanced, 45, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    candidates.append(edges)

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        7,
    )
    adaptive = cv2.bitwise_not(adaptive)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    candidates.append(adaptive)

    min_area = 0.20 * image.shape[0] * image.shape[1]

    for mask in candidates:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue

            area = cv2.contourArea(approx)
            if area < min_area:
                continue

            quad = approx.reshape(4, 2).astype("float32")
            rect = _order_points(quad)
            (tl, tr, br, bl) = rect

            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            max_width = max(width_top, width_bottom)

            height_right = np.linalg.norm(br - tr)
            height_left = np.linalg.norm(bl - tl)
            max_height = max(height_right, height_left)

            if max_width < 20 or max_height < 20:
                continue

            ratio = max_height / max_width
            if expected_ratio is not None and not _is_ratio_close(ratio, expected_ratio):
                continue

            return rect

    return None


def _warp_from_quad(image: np.ndarray, rect: np.ndarray, force_ratio: float | None = None) -> np.ndarray:
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = int(max(height_right, height_left))

    if force_ratio is not None and max_width > 0:
        max_height = int(max_width * force_ratio)

    max_width = max(1, max_width)
    max_height = max(1, max_height)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _scan_like_camscanner(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    quad = _find_document_quad(frame, expected_ratio=OMR_RATIO)
    if quad is not None:
        return _warp_from_quad(frame, quad, force_ratio=OMR_RATIO), True

    h, w, _ = frame.shape
    x1, y1, x2, y2 = _safe_center_box(w, h)
    center_crop = frame[y1:y2, x1:x2]

    quad_center = _find_document_quad(center_crop, expected_ratio=OMR_RATIO)
    if quad_center is not None:
        return _warp_from_quad(center_crop, quad_center, force_ratio=OMR_RATIO), True

    return center_crop, False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img")
    if not data:
        return jsonify({"detected": False}), 200

    img = _decode_data_url_image(data)
    if img is None:
        return jsonify({"detected": False}), 200

    quad = _find_document_quad(img, expected_ratio=OMR_RATIO)
    detected = quad is not None

    return jsonify({"detected": detected})


@app.route("/process", methods=["POST"])
def process():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img") or request.form.get("img")
    if not data:
        return "no image", 400

    img = _decode_data_url_image(data)
    if img is None:
        return "decode failed", 400

    scanned, _ = _scan_like_camscanner(img)

    oh, ow, _ = scanned.shape
    ax = int(ow * X_RATIO)
    ay = int(oh * Y_RATIO)
    aw = int(ow * W_RATIO)
    ah = int(oh * H_RATIO)

    ax = max(0, ax)
    ay = max(0, ay)
    aw = max(1, min(ow - ax, aw))
    ah = max(1, min(oh - ay, ah))

    answer = scanned[ay : ay + ah, ax : ax + aw]

    ok, buf = cv2.imencode(".jpg", answer)
    if not ok:
        return "encode failed", 400

    return send_file(BytesIO(buf), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)






