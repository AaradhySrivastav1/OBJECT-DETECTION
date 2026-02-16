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


def _default_quad_for_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")


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




def _crop_biggest_rectangle(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        7,
    )
    th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    edges = cv2.Canny(blurred, 40, 140)
    merged = cv2.bitwise_or(th, edges)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    min_area = 0.20 * image.shape[0] * image.shape[1]
    best_quad = None
    best_area = 0.0

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
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            quad = _order_points(box.astype("float32"))

        w_top = np.linalg.norm(quad[1] - quad[0])
        w_bottom = np.linalg.norm(quad[2] - quad[3])
        h_right = np.linalg.norm(quad[2] - quad[1])
        h_left = np.linalg.norm(quad[3] - quad[0])

        rect_width = max(w_top, w_bottom)
        rect_height = max(h_right, h_left)
        if rect_width < 5 or rect_height < 5:
            continue

        ratio = rect_height / rect_width
        if abs(ratio - OMR_RATIO) > OMR_RATIO * 0.35:
            continue

        if area > best_area:
            best_area = area
            best_quad = quad

    if best_quad is not None:
        doc = four_point_transform(image, best_quad)
        return doc, merged, best_quad

    return image, merged, None

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
    captured_doc: np.ndarray,
    edges: np.ndarray,
    quad: np.ndarray | None,
    pre_crop_vis: np.ndarray,
) -> np.ndarray:
    roi_vis = captured_doc.copy()
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

    cv2.putText(left, "Captured Document", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)
    cv2.putText(mid, "Threshold + Biggest Rect", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(right, "Final Before Crop", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 180, 50), 2)

    return cv2.hconcat([left, mid, right])


def process_frame(frame: np.ndarray) -> dict:
    roi = _extract_box_region(frame)
    quad, _, edges = detect_document(roi)

    stable = stability_check(quad)
    detected = quad is not None

    warped = four_point_transform(roi, quad) if detected else cv2.resize(roi, (WARP_WIDTH, WARP_HEIGHT), interpolation=cv2.INTER_LINEAR)
    captured_doc, doc_edges, doc_quad = _crop_biggest_rectangle(warped)
    captured_doc = cv2.resize(captured_doc, (WARP_WIDTH, WARP_HEIGHT), interpolation=cv2.INTER_LINEAR)
    processed = apply_lighting_normalization(captured_doc)

    return {
        "roi": roi,
        "edges": edges,
        "quad": quad,
        "stable": stable,
        "detected": detected,
        "can_capture": detected,
        "warped": warped,
        "captured_doc": captured_doc,
        "doc_edges": doc_edges,
        "doc_quad": doc_quad,
        "processed": processed,
    }


def _process_from_manual_quad(roi: np.ndarray, quad: np.ndarray) -> dict:
    warped = four_point_transform(roi, quad)
    captured_doc, doc_edges, doc_quad = _crop_biggest_rectangle(warped)
    captured_doc = cv2.resize(captured_doc, (WARP_WIDTH, WARP_HEIGHT), interpolation=cv2.INTER_LINEAR)
    processed = apply_lighting_normalization(captured_doc)

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

    pre_crop_vis = captured_doc.copy()
    cv2.rectangle(pre_crop_vis, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 3)
    pipeline_preview = _make_pipeline_preview(captured_doc, doc_edges, doc_quad, pre_crop_vis)

    captured_url = _encode_image_data_url(roi, quality=92)
    pipeline_url = _encode_image_data_url(pipeline_preview, quality=92)
    biggest_rect_url = _encode_image_data_url(captured_doc, quality=92)
    cropped_url = _encode_image_data_url(answer, quality=92)

    if not captured_url or not pipeline_url or not biggest_rect_url or not cropped_url:
        raise ValueError("encode failed")

    return {
        "captured_image": captured_url,
        "pipeline_image": pipeline_url,
        "biggest_rect_image": biggest_rect_url,
        "cropped_image": cropped_url,
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
    try:
        output = _process_from_manual_quad(frame_info["roi"], frame_info["quad"] if frame_info["quad"] is not None else _default_quad_for_image(frame_info["roi"]))
    except ValueError:
        return "encode failed", 400

    output["stable"] = frame_info["stable"]
    return jsonify(output)


@app.route("/prepare", methods=["POST"])
def prepare():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img")
    if not data:
        return "no image", 400

    img = _decode_data_url_image(data)
    if img is None:
        return "decode failed", 400

    roi = _extract_box_region(img)
    quad, _, _ = detect_document(roi)
    if quad is None:
        quad = _default_quad_for_image(roi)

    roi_url = _encode_image_data_url(roi, quality=92)
    if not roi_url:
        return "encode failed", 400

    return jsonify({"editor_image": roi_url, "quad": quad.tolist()})


@app.route("/process_manual", methods=["POST"])
def process_manual():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img")
    points = payload.get("points")
    if not data or not points:
        return "invalid payload", 400

    roi = _decode_data_url_image(data)
    if roi is None:
        return "decode failed", 400

    try:
        quad = np.array(points, dtype="float32")
    except (TypeError, ValueError):
        return "invalid points", 400

    if quad.shape != (4, 2):
        return "invalid points", 400

    try:
        output = _process_from_manual_quad(roi, quad)
    except ValueError:
        return "encode failed", 400

    output["stable"] = True
    return jsonify(output)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

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
#     edges = cv2.Canny(blurred, 40, 140)
#     edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

#     adaptive = cv2.adaptiveThreshold(
#         gray,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         21,
#         7,
#     )
#     adaptive = cv2.bitwise_not(adaptive)
#     adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

#     merged_edges = cv2.bitwise_or(edges, adaptive)

#     contours, _ = cv2.findContours(merged_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)

#     min_area = MIN_DOC_AREA_RATIO * frame.shape[0] * frame.shape[1]
#     for contour in contours[:30]:
#         area = cv2.contourArea(contour)
#         if area < min_area:
#             continue

#         perimeter = cv2.arcLength(contour, True)
#         if perimeter <= 0:
#             continue

#         approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
#         if len(approx) == 4:
#             quad = _order_points(approx.reshape(4, 2).astype("float32"))
#             return quad, gray, merged_edges

#     # fallback to minimum-area rectangle of the largest contour
#     for contour in contours[:10]:
#         area = cv2.contourArea(contour)
#         if area < min_area:
#             continue

#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         quad = _order_points(box.astype("float32"))
#         return quad, gray, merged_edges

#     return None, gray, merged_edges


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




# def _crop_biggest_rectangle(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     th = cv2.adaptiveThreshold(
#         blurred,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         21,
#         7,
#     )
#     th = cv2.bitwise_not(th)
#     th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

#     edges = cv2.Canny(blurred, 40, 140)
#     merged = cv2.bitwise_or(th, edges)

#     contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)

#     min_area = 0.20 * image.shape[0] * image.shape[1]
#     best_quad = None
#     best_area = 0.0

#     for contour in contours[:30]:
#         area = cv2.contourArea(contour)
#         if area < min_area:
#             continue

#         perimeter = cv2.arcLength(contour, True)
#         if perimeter <= 0:
#             continue

#         approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
#         if len(approx) == 4:
#             quad = _order_points(approx.reshape(4, 2).astype("float32"))
#         else:
#             rect = cv2.minAreaRect(contour)
#             box = cv2.boxPoints(rect)
#             quad = _order_points(box.astype("float32"))

#         w_top = np.linalg.norm(quad[1] - quad[0])
#         w_bottom = np.linalg.norm(quad[2] - quad[3])
#         h_right = np.linalg.norm(quad[2] - quad[1])
#         h_left = np.linalg.norm(quad[3] - quad[0])

#         rect_width = max(w_top, w_bottom)
#         rect_height = max(h_right, h_left)
#         if rect_width < 5 or rect_height < 5:
#             continue

#         ratio = rect_height / rect_width
#         if abs(ratio - OMR_RATIO) > OMR_RATIO * 0.35:
#             continue

#         if area > best_area:
#             best_area = area
#             best_quad = quad

#     if best_quad is not None:
#         doc = four_point_transform(image, best_quad)
#         return doc, merged, best_quad

#     return image, merged, None

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
#     captured_doc: np.ndarray,
#     edges: np.ndarray,
#     quad: np.ndarray | None,
#     pre_crop_vis: np.ndarray,
# ) -> np.ndarray:
#     roi_vis = captured_doc.copy()
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

#     cv2.putText(left, "Captured Document", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)
#     cv2.putText(mid, "Threshold + Biggest Rect", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
#     cv2.putText(right, "Final Before Crop", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 180, 50), 2)

#     return cv2.hconcat([left, mid, right])


# def process_frame(frame: np.ndarray) -> dict:
#     roi = _extract_box_region(frame)
#     quad, _, edges = detect_document(roi)

#     stable = stability_check(quad)
#     detected = quad is not None

#     warped = four_point_transform(roi, quad) if detected else cv2.resize(roi, (WARP_WIDTH, WARP_HEIGHT), interpolation=cv2.INTER_LINEAR)
#     captured_doc, doc_edges, doc_quad = _crop_biggest_rectangle(warped)
#     captured_doc = cv2.resize(captured_doc, (WARP_WIDTH, WARP_HEIGHT), interpolation=cv2.INTER_LINEAR)
#     processed = apply_lighting_normalization(captured_doc)

#     return {
#         "roi": roi,
#         "edges": edges,
#         "quad": quad,
#         "stable": stable,
#         "detected": detected,
#         "can_capture": detected,
#         "warped": warped,
#         "captured_doc": captured_doc,
#         "doc_edges": doc_edges,
#         "doc_quad": doc_quad,
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

#     pre_crop_vis = frame_info["captured_doc"].copy()
#     cv2.rectangle(pre_crop_vis, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 3)

#     pipeline_preview = _make_pipeline_preview(frame_info["captured_doc"], frame_info["doc_edges"], frame_info["doc_quad"], pre_crop_vis)

#     captured_url = _encode_image_data_url(frame_info["roi"], quality=92)
#     pipeline_url = _encode_image_data_url(pipeline_preview, quality=92)
#     biggest_rect_url = _encode_image_data_url(frame_info["captured_doc"], quality=92)
#     cropped_url = _encode_image_data_url(answer, quality=92)

#     if not captured_url or not pipeline_url or not biggest_rect_url or not cropped_url:
#         return "encode failed", 400

#     return jsonify(
#         {
#             "captured_image": captured_url,
#             "pipeline_image": pipeline_url,
#             "biggest_rect_image": biggest_rect_url,
#             "cropped_image": cropped_url,
#             "stable": frame_info["stable"],
#         }
#     )


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=10000)

