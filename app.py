from flask import Flask, render_template, request, jsonify
import base64

import cv2
import numpy as np

app = Flask(__name__)

OMR_RATIO = 26.5 / 21.5
BOX_WIDTH_RATIO = 0.84
X_RATIO = 0.005
Y_RATIO = 0.57
W_RATIO = 0.99
H_RATIO = 0.22

STABILITY_REQUIRED_FRAMES = 6
SHAPE_DIFF_THRESHOLD = 0.03
MIN_DOC_AREA_RATIO = 0.12
WARP_WIDTH = 900
WARP_HEIGHT = int(WARP_WIDTH * OMR_RATIO)

stability_state = {
    "last_contour": None,
    "count": 0,
    "stable": False,
}


def _decode_data_url_image(data: str) -> np.ndarray | None:
    _, _, encoded = data.partition(",")
    if not encoded:
        return None

    try:
        img_bytes = base64.b64decode(encoded)
    except (base64.binascii.Error, ValueError):
        return None

    return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)


def _encode_image_data_url(image: np.ndarray, quality: int = 90) -> str | None:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode(".jpg", image, encode_params)
    if not ok:
        return None

    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _omr_box_coords(width: int, height: int) -> tuple[int, int, int, int]:
    max_width_by_height = int(height / OMR_RATIO)
    box_width = min(int(width * BOX_WIDTH_RATIO), max_width_by_height)
    box_height = int(box_width * OMR_RATIO)

    x1 = max(0, (width - box_width) // 2)
    y1 = max(0, (height - box_height) // 2)
    x2 = min(width, x1 + box_width)
    y2 = min(height, y1 + box_height)
    return x1, y1, x2, y2


def _extract_box_region(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _omr_box_coords(w, h)
    return frame[y1:y2, x1:x2]


def detect_document(frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        7,
    )
    adaptive = cv2.bitwise_not(adaptive)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    merged_edges = cv2.bitwise_or(edges, adaptive)

    contours, _ = cv2.findContours(merged_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    min_area = MIN_DOC_AREA_RATIO * frame.shape[0] * frame.shape[1]
    for contour in contours[:30]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            quad = _order_points(approx.reshape(4, 2).astype("float32"))
            return quad, gray, merged_edges

    # fallback to minimum-area rectangle of the largest contour
    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        quad = _order_points(box.astype("float32"))
        return quad, gray, merged_edges

    return None, gray, merged_edges


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    dst = np.array(
        [[0, 0], [WARP_WIDTH - 1, 0], [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (WARP_WIDTH, WARP_HEIGHT))


def apply_lighting_normalization(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    return cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        8,
    )


def stability_check(current_contour: np.ndarray | None) -> bool:
    if current_contour is None:
        stability_state["last_contour"] = None
        stability_state["count"] = 0
        stability_state["stable"] = False
        return False

    if stability_state["last_contour"] is None:
        stability_state["last_contour"] = current_contour.copy()
        stability_state["count"] = 1
        stability_state["stable"] = False
        return False

    diff = cv2.matchShapes(stability_state["last_contour"], current_contour, cv2.CONTOURS_MATCH_I1, 0.0)
    stability_state["count"] = stability_state["count"] + 1 if diff < SHAPE_DIFF_THRESHOLD else 1

    stability_state["last_contour"] = current_contour.copy()
    stability_state["stable"] = stability_state["count"] >= STABILITY_REQUIRED_FRAMES
    return stability_state["stable"]


def _make_pipeline_preview(
    roi: np.ndarray,
    edges: np.ndarray,
    quad: np.ndarray | None,
    pre_crop_vis: np.ndarray,
) -> np.ndarray:
    roi_vis = roi.copy()
    if quad is not None:
        cv2.polylines(roi_vis, [quad.astype(np.int32)], True, (0, 255, 0), 3)

    edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    h_target = 320

    def _resize_keep(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        nw = max(1, int(w * (h_target / h)))
        return cv2.resize(img, (nw, h_target), interpolation=cv2.INTER_AREA)

    left = _resize_keep(roi_vis)
    mid = _resize_keep(edges_vis)
    right = _resize_keep(pre_crop_vis)

    cv2.putText(left, "Captured Box", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)
    cv2.putText(mid, "Edge + Shape", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(right, "Final Before Crop", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 180, 50), 2)

    return cv2.hconcat([left, mid, right])


def process_frame(frame: np.ndarray) -> dict:
    roi = _extract_box_region(frame)
    quad, _, edges = detect_document(roi)

    stable = stability_check(quad)
    detected = quad is not None

    warped = four_point_transform(roi, quad) if detected else cv2.resize(roi, (WARP_WIDTH, WARP_HEIGHT), interpolation=cv2.INTER_LINEAR)
    processed = apply_lighting_normalization(warped)

    return {
        "roi": roi,
        "edges": edges,
        "quad": quad,
        "stable": stable,
        "detected": detected,
        "can_capture": detected,
        "warped": warped,
        "processed": processed,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img")
    if not data:
        return jsonify({"detected": False, "stable": False, "can_capture": False}), 200

    img = _decode_data_url_image(data)
    if img is None:
        return jsonify({"detected": False, "stable": False, "can_capture": False}), 200

    frame_info = process_frame(img)
    return jsonify(
        {
            "detected": frame_info["detected"],
            "stable": frame_info["stable"],
            "can_capture": frame_info["can_capture"],
        }
    )


@app.route("/process", methods=["POST"])
def process():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img") or request.form.get("img")
    if not data:
        return "no image", 400

    img = _decode_data_url_image(data)
    if img is None:
        return "decode failed", 400

    frame_info = process_frame(img)
    processed = frame_info["processed"]

    oh, ow = processed.shape[:2]
    ax = int(ow * X_RATIO)
    ay = int(oh * Y_RATIO)
    aw = int(ow * W_RATIO)
    ah = int(oh * H_RATIO)

    ax = max(0, ax)
    ay = max(0, ay)
    aw = max(1, min(ow - ax, aw))
    ah = max(1, min(oh - ay, ah))

    answer = processed[ay : ay + ah, ax : ax + aw]

    pre_crop_vis = frame_info["warped"].copy()
    cv2.rectangle(pre_crop_vis, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 3)

    pipeline_preview = _make_pipeline_preview(frame_info["roi"], frame_info["edges"], frame_info["quad"], pre_crop_vis)

    captured_url = _encode_image_data_url(frame_info["roi"], quality=92)
    pipeline_url = _encode_image_data_url(pipeline_preview, quality=92)
    cropped_url = _encode_image_data_url(answer, quality=92)

    if not captured_url or not pipeline_url or not cropped_url:
        return "encode failed", 400

    return jsonify(
        {
            "captured_image": captured_url,
            "pipeline_image": pipeline_url,
            "cropped_image": cropped_url,
            "stable": frame_info["stable"],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)









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

# from flask import Flask, render_template, request, send_file, jsonify
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


# def _decode_data_url_image(data: str) -> np.ndarray | None:
#     _, _, encoded = data.partition(",")
#     if not encoded:
#         return None

#     try:
#         img_bytes = base64.b64decode(encoded)
#     except (base64.binascii.Error, ValueError):
#         return None

#     return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)


# def _is_ratio_close(ratio: float, expected: float, tolerance: float = 0.40) -> bool:
#     return abs(ratio - expected) <= expected * tolerance


# def _find_document_quad(image: np.ndarray, expected_ratio: float | None = None) -> np.ndarray | None:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)

#     candidates = []

#     edges = cv2.Canny(enhanced, 45, 150)
#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
#     candidates.append(edges)

#     adaptive = cv2.adaptiveThreshold(
#         enhanced,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         21,
#         7,
#     )
#     adaptive = cv2.bitwise_not(adaptive)
#     adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
#     candidates.append(adaptive)

#     min_area = 0.20 * image.shape[0] * image.shape[1]

#     for mask in candidates:
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#         for contour in contours:
#             perimeter = cv2.arcLength(contour, True)
#             if perimeter <= 0:
#                 continue

#             approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
#             if len(approx) != 4:
#                 continue

#             area = cv2.contourArea(approx)
#             if area < min_area:
#                 continue

#             quad = approx.reshape(4, 2).astype("float32")
#             rect = _order_points(quad)
#             (tl, tr, br, bl) = rect

#             width_top = np.linalg.norm(tr - tl)
#             width_bottom = np.linalg.norm(br - bl)
#             max_width = max(width_top, width_bottom)

#             height_right = np.linalg.norm(br - tr)
#             height_left = np.linalg.norm(bl - tl)
#             max_height = max(height_right, height_left)

#             if max_width < 20 or max_height < 20:
#                 continue

#             ratio = max_height / max_width
#             if expected_ratio is not None and not _is_ratio_close(ratio, expected_ratio):
#                 continue

#             return rect

#     return None


# def _warp_from_quad(image: np.ndarray, rect: np.ndarray, force_ratio: float | None = None) -> np.ndarray:
#     (tl, tr, br, bl) = rect

#     width_top = np.linalg.norm(tr - tl)
#     width_bottom = np.linalg.norm(br - bl)
#     max_width = int(max(width_top, width_bottom))

#     height_right = np.linalg.norm(br - tr)
#     height_left = np.linalg.norm(bl - tl)
#     max_height = int(max(height_right, height_left))

#     if force_ratio is not None and max_width > 0:
#         max_height = int(max_width * force_ratio)

#     max_width = max(1, max_width)
#     max_height = max(1, max_height)

#     dst = np.array(
#         [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
#         dtype="float32",
#     )

#     matrix = cv2.getPerspectiveTransform(rect, dst)
#     return cv2.warpPerspective(image, matrix, (max_width, max_height))


# def _scan_like_camscanner(frame: np.ndarray) -> tuple[np.ndarray, bool]:
#     quad = _find_document_quad(frame, expected_ratio=OMR_RATIO)
#     if quad is not None:
#         return _warp_from_quad(frame, quad, force_ratio=OMR_RATIO), True

#     h, w, _ = frame.shape
#     x1, y1, x2, y2 = _safe_center_box(w, h)
#     center_crop = frame[y1:y2, x1:x2]

#     quad_center = _find_document_quad(center_crop, expected_ratio=OMR_RATIO)
#     if quad_center is not None:
#         return _warp_from_quad(center_crop, quad_center, force_ratio=OMR_RATIO), True

#     return center_crop, False


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/detect", methods=["POST"])
# def detect():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("img")
#     if not data:
#         return jsonify({"detected": False}), 200

#     img = _decode_data_url_image(data)
#     if img is None:
#         return jsonify({"detected": False}), 200

#     quad = _find_document_quad(img, expected_ratio=OMR_RATIO)
#     detected = quad is not None

#     return jsonify({"detected": detected})


# @app.route("/process", methods=["POST"])
# def process():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("img") or request.form.get("img")
#     if not data:
#         return "no image", 400

#     img = _decode_data_url_image(data)
#     if img is None:
#         return "decode failed", 400

#     scanned, _ = _scan_like_camscanner(img)

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



# from flask import Flask, render_template, request, send_file, jsonify
# import base64
# from io import BytesIO

# import cv2
# import numpy as np

# app = Flask(__name__)

# OMR_RATIO = 26.5 / 21.5
# X_RATIO = 0.005
# Y_RATIO = 0.57
# W_RATIO = 0.99
# H_RATIO = 0.22

# STABILITY_REQUIRED_FRAMES = 15
# SHAPE_DIFF_THRESHOLD = 0.02
# MIN_DOC_AREA_RATIO = 0.18
# WARP_WIDTH = 900
# WARP_HEIGHT = 1200

# stability_state = {
#     "last_contour": None,
#     "count": 0,
#     "stable": False,
# }


# def _decode_data_url_image(data: str) -> np.ndarray | None:
#     _, _, encoded = data.partition(",")
#     if not encoded:
#         return None

#     try:
#         img_bytes = base64.b64decode(encoded)
#     except (base64.binascii.Error, ValueError):
#         return None

#     return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)


# def _order_points(points: np.ndarray) -> np.ndarray:
#     rect = np.zeros((4, 2), dtype="float32")
#     s = points.sum(axis=1)
#     diff = np.diff(points, axis=1)

#     rect[0] = points[np.argmin(s)]
#     rect[2] = points[np.argmax(s)]
#     rect[1] = points[np.argmin(diff)]
#     rect[3] = points[np.argmax(diff)]

#     return rect


# def detect_document(frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     edges = cv2.Canny(blurred, 50, 150)
#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)

#     min_area = MIN_DOC_AREA_RATIO * frame.shape[0] * frame.shape[1]
#     for contour in contours[:20]:
#         perimeter = cv2.arcLength(contour, True)
#         if perimeter <= 0:
#             continue

#         approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
#         if len(approx) != 4:
#             continue

#         area = cv2.contourArea(approx)
#         if area < min_area:
#             continue

#         quad = approx.reshape(4, 2).astype("float32")
#         quad = _order_points(quad)

#         w_top = np.linalg.norm(quad[1] - quad[0])
#         w_bottom = np.linalg.norm(quad[2] - quad[3])
#         h_right = np.linalg.norm(quad[2] - quad[1])
#         h_left = np.linalg.norm(quad[3] - quad[0])

#         doc_ratio = max(h_left, h_right) / max(max(w_top, w_bottom), 1.0)
#         if abs(doc_ratio - OMR_RATIO) > OMR_RATIO * 0.45:
#             continue

#         return quad, gray, edges

#     return None, gray, edges


# def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
#     rect = _order_points(pts)
#     (tl, tr, br, bl) = rect

#     dst = np.array(
#         [[0, 0], [WARP_WIDTH - 1, 0], [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]],
#         dtype="float32",
#     )

#     matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
#     return cv2.warpPerspective(image, matrix, (WARP_WIDTH, WARP_HEIGHT))


# def apply_lighting_normalization(image: np.ndarray) -> np.ndarray:
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image

#     clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
#     normalized = clahe.apply(gray)

#     binary = cv2.adaptiveThreshold(
#         normalized,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         21,
#         8,
#     )
#     return binary


# def stability_check(current_contour: np.ndarray | None) -> bool:
#     if current_contour is None:
#         stability_state["last_contour"] = None
#         stability_state["count"] = 0
#         stability_state["stable"] = False
#         return False

#     if stability_state["last_contour"] is None:
#         stability_state["last_contour"] = current_contour.copy()
#         stability_state["count"] = 1
#         stability_state["stable"] = False
#         return False

#     diff = cv2.matchShapes(stability_state["last_contour"], current_contour, cv2.CONTOURS_MATCH_I1, 0.0)

#     if diff < SHAPE_DIFF_THRESHOLD:
#         stability_state["count"] += 1
#     else:
#         stability_state["count"] = 1

#     stability_state["last_contour"] = current_contour.copy()
#     stability_state["stable"] = stability_state["count"] >= STABILITY_REQUIRED_FRAMES
#     return stability_state["stable"]


# def process_frame(frame: np.ndarray) -> dict:
#     quad, gray, edges = detect_document(frame)

#     stable = stability_check(quad)
#     can_capture = quad is not None and stable

#     annotated = frame.copy()
#     if quad is not None:
#         cv2.polylines(annotated, [quad.astype(np.int32)], True, (0, 255, 0) if stable else (0, 0, 255), 3)

#     if quad is not None:
#         warped = four_point_transform(frame, quad)
#     else:
#         warped = cv2.resize(frame, (WARP_WIDTH, WARP_HEIGHT), interpolation=cv2.INTER_LINEAR)

#     normalized_binary = apply_lighting_normalization(warped)

#     return {
#         "gray": gray,
#         "edges": edges,
#         "quad": quad,
#         "stable": stable,
#         "can_capture": can_capture,
#         "annotated": annotated,
#         "warped": warped,
#         "processed": normalized_binary,
#     }


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/detect", methods=["POST"])
# def detect():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("img")
#     if not data:
#         return jsonify({"detected": False, "stable": False, "can_capture": False}), 200

#     img = _decode_data_url_image(data)
#     if img is None:
#         return jsonify({"detected": False, "stable": False, "can_capture": False}), 200

#     frame_info = process_frame(img)
#     quad = frame_info["quad"]

#     return jsonify(
#         {
#             "detected": quad is not None,
#             "stable": frame_info["stable"],
#             "can_capture": frame_info["can_capture"],
#         }
#     )


# @app.route("/process", methods=["POST"])
# def process():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("img") or request.form.get("img")
#     if not data:
#         return "no image", 400

#     img = _decode_data_url_image(data)
#     if img is None:
#         return "decode failed", 400

#     frame_info = process_frame(img)
#     processed = frame_info["processed"]

#     oh, ow = processed.shape[:2]
#     ax = int(ow * X_RATIO)
#     ay = int(oh * Y_RATIO)
#     aw = int(ow * W_RATIO)
#     ah = int(oh * H_RATIO)

#     ax = max(0, ax)
#     ay = max(0, ay)
#     aw = max(1, min(ow - ax, aw))
#     ah = max(1, min(oh - ay, ah))

#     answer = processed[ay : ay + ah, ax : ax + aw]

#     ok, buf = cv2.imencode(".jpg", answer)
#     if not ok:
#         return "encode failed", 400

#     return send_file(BytesIO(buf), mimetype="image/jpeg")


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=10000)
# from flask import Flask, render_template, request, jsonify
# import base64

# import cv2
# import numpy as np

# app = Flask(__name__)

# OMR_RATIO = 26.5 / 21.5
# BOX_WIDTH_RATIO = 0.84
# X_RATIO = 0.005
# Y_RATIO = 0.57
# W_RATIO = 0.99
# H_RATIO = 0.22

# STABILITY_REQUIRED_FRAMES = 6
# SHAPE_DIFF_THRESHOLD = 0.03
# MIN_DOC_AREA_RATIO = 0.12
# WARP_WIDTH = 900
# WARP_HEIGHT = int(WARP_WIDTH * OMR_RATIO)

# stability_state = {
#     "last_contour": None,
#     "count": 0,
#     "stable": False,
# }


# def _decode_data_url_image(data: str) -> np.ndarray | None:
#     _, _, encoded = data.partition(",")
#     if not encoded:
#         return None

#     try:
#         img_bytes = base64.b64decode(encoded)
#     except (base64.binascii.Error, ValueError):
#         return None

#     return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)


# def _encode_image_data_url(image: np.ndarray, quality: int = 90) -> str | None:
#     encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
#     ok, buf = cv2.imencode(".jpg", image, encode_params)
#     if not ok:
#         return None

#     return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


# def _order_points(points: np.ndarray) -> np.ndarray:
#     rect = np.zeros((4, 2), dtype="float32")
#     s = points.sum(axis=1)
#     diff = np.diff(points, axis=1)

#     rect[0] = points[np.argmin(s)]
#     rect[2] = points[np.argmax(s)]
#     rect[1] = points[np.argmin(diff)]
#     rect[3] = points[np.argmax(diff)]
#     return rect


# def _omr_box_coords(width: int, height: int) -> tuple[int, int, int, int]:
#     max_width_by_height = int(height / OMR_RATIO)
#     box_width = min(int(width * BOX_WIDTH_RATIO), max_width_by_height)
#     box_height = int(box_width * OMR_RATIO)

#     x1 = max(0, (width - box_width) // 2)
#     y1 = max(0, (height - box_height) // 2)
#     x2 = min(width, x1 + box_width)
#     y2 = min(height, y1 + box_height)
#     return x1, y1, x2, y2


# def _extract_box_region(frame: np.ndarray) -> np.ndarray:
#     h, w = frame.shape[:2]
#     x1, y1, x2, y2 = _omr_box_coords(w, h)
#     return frame[y1:y2, x1:x2]


# def detect_document(frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)

#     min_area = MIN_DOC_AREA_RATIO * frame.shape[0] * frame.shape[1]
#     for contour in contours[:20]:
#         perimeter = cv2.arcLength(contour, True)
#         if perimeter <= 0:
#             continue

#         approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
#         if len(approx) != 4:
#             continue

#         area = cv2.contourArea(approx)
#         if area < min_area:
#             continue

#         quad = _order_points(approx.reshape(4, 2).astype("float32"))

#         w_top = np.linalg.norm(quad[1] - quad[0])
#         w_bottom = np.linalg.norm(quad[2] - quad[3])
#         h_right = np.linalg.norm(quad[2] - quad[1])
#         h_left = np.linalg.norm(quad[3] - quad[0])

#         doc_ratio = max(h_left, h_right) / max(max(w_top, w_bottom), 1.0)
#         if abs(doc_ratio - OMR_RATIO) > OMR_RATIO * 0.5:
#             continue

#         return quad, gray, edges

#     return None, gray, edges


# def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
#     rect = _order_points(pts)
#     dst = np.array(
#         [[0, 0], [WARP_WIDTH - 1, 0], [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]],
#         dtype="float32",
#     )
#     matrix = cv2.getPerspectiveTransform(rect, dst)
#     return cv2.warpPerspective(image, matrix, (WARP_WIDTH, WARP_HEIGHT))


# def apply_lighting_normalization(image: np.ndarray) -> np.ndarray:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
#     clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
#     normalized = clahe.apply(gray)
#     return cv2.adaptiveThreshold(
#         normalized,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         21,
#         8,
#     )


# def stability_check(current_contour: np.ndarray | None) -> bool:
#     if current_contour is None:
#         stability_state["last_contour"] = None
#         stability_state["count"] = 0
#         stability_state["stable"] = False
#         return False

#     if stability_state["last_contour"] is None:
#         stability_state["last_contour"] = current_contour.copy()
#         stability_state["count"] = 1
#         stability_state["stable"] = False
#         return False

#     diff = cv2.matchShapes(stability_state["last_contour"], current_contour, cv2.CONTOURS_MATCH_I1, 0.0)
#     stability_state["count"] = stability_state["count"] + 1 if diff < SHAPE_DIFF_THRESHOLD else 1

#     stability_state["last_contour"] = current_contour.copy()
#     stability_state["stable"] = stability_state["count"] >= STABILITY_REQUIRED_FRAMES
#     return stability_state["stable"]


# def _make_pipeline_preview(
#     roi: np.ndarray,
#     edges: np.ndarray,
#     quad: np.ndarray | None,
#     pre_crop_vis: np.ndarray,
# ) -> np.ndarray:
#     roi_vis = roi.copy()
#     if quad is not None:
#         cv2.polylines(roi_vis, [quad.astype(np.int32)], True, (0, 255, 0), 3)

#     edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     h_target = 320

#     def _resize_keep(img: np.ndarray) -> np.ndarray:
#         h, w = img.shape[:2]
#         nw = max(1, int(w * (h_target / h)))
#         return cv2.resize(img, (nw, h_target), interpolation=cv2.INTER_AREA)

#     left = _resize_keep(roi_vis)
#     mid = _resize_keep(edges_vis)
#     right = _resize_keep(pre_crop_vis)

#     cv2.putText(left, "Captured Box", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)
#     cv2.putText(mid, "Edge + Shape", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#     cv2.putText(right, "Final Before Crop", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 180, 50), 2)

#     return cv2.hconcat([left, mid, right])


# def process_frame(frame: np.ndarray) -> dict:
#     roi = _extract_box_region(frame)
#     quad, _, edges = detect_document(roi)

#     stable = stability_check(quad)
#     detected = quad is not None

#     warped = four_point_transform(roi, quad) if detected else cv2.resize(roi, (WARP_WIDTH, WARP_HEIGHT), interpolation=cv2.INTER_LINEAR)
#     processed = apply_lighting_normalization(warped)

#     return {
#         "roi": roi,
#         "edges": edges,
#         "quad": quad,
#         "stable": stable,
#         "detected": detected,
#         "can_capture": detected,
#         "processed": processed,
#     }


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/detect", methods=["POST"])
# def detect():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("img")
#     if not data:
#         return jsonify({"detected": False, "stable": False, "can_capture": False}), 200

#     img = _decode_data_url_image(data)
#     if img is None:
#         return jsonify({"detected": False, "stable": False, "can_capture": False}), 200

#     frame_info = process_frame(img)
#     return jsonify(
#         {
#             "detected": frame_info["detected"],
#             "stable": frame_info["stable"],
#             "can_capture": frame_info["can_capture"],
#         }
#     )


# @app.route("/process", methods=["POST"])
# def process():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("img") or request.form.get("img")
#     if not data:
#         return "no image", 400

#     img = _decode_data_url_image(data)
#     if img is None:
#         return "decode failed", 400

#     frame_info = process_frame(img)
#     processed = frame_info["processed"]

#     oh, ow = processed.shape[:2]
#     ax = int(ow * X_RATIO)
#     ay = int(oh * Y_RATIO)
#     aw = int(ow * W_RATIO)
#     ah = int(oh * H_RATIO)

#     ax = max(0, ax)
#     ay = max(0, ay)
#     aw = max(1, min(ow - ax, aw))
#     ah = max(1, min(oh - ay, ah))

#     answer = processed[ay : ay + ah, ax : ax + aw]

#     pre_crop_vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
#     cv2.rectangle(pre_crop_vis, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 3)

#     pipeline_preview = _make_pipeline_preview(frame_info["roi"], frame_info["edges"], frame_info["quad"], pre_crop_vis)

#     captured_url = _encode_image_data_url(frame_info["roi"], quality=92)
#     pipeline_url = _encode_image_data_url(pipeline_preview, quality=92)
#     cropped_url = _encode_image_data_url(answer, quality=92)

#     if not captured_url or not pipeline_url or not cropped_url:
#         return "encode failed", 400

#     return jsonify(
#         {
#             "captured_image": captured_url,
#             "pipeline_image": pipeline_url,
#             "cropped_image": cropped_url,
#             "stable": frame_info["stable"],
#         }
#     )


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=10000)

