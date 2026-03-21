from __future__ import annotations

from flask import Flask, jsonify, render_template, request
import base64
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

app = Flask(__name__)

OMR_RATIO = 27.0 / 21.0
BOX_WIDTH_RATIO = 0.84
WARP_WIDTH = 900
WARP_HEIGHT = int(WARP_WIDTH * OMR_RATIO)
MIN_DOC_AREA_RATIO = 0.12
ANSWER_ROWS = 5
ANSWER_COLS = 4
ANSWER_RATIO_FALLBACK = (0.06, 0.56, 0.88, 0.30)  # x, y, w, h on scanned image


def _decode_data_url_image(data: str) -> np.ndarray | None:
    _, _, encoded = data.partition(",")
    if not encoded:
        return None

    try:
        img_bytes = base64.b64decode(encoded)
    except (base64.binascii.Error, ValueError):
        return None

    return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)


def _encode_image_data_url(image: np.ndarray, quality: int = 92) -> str | None:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
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


def _default_quad_for_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")


def _omr_box_coords(width: int, height: int) -> tuple[int, int, int, int]:
    max_width_by_height = int(height / OMR_RATIO)
    box_width = min(int(width * BOX_WIDTH_RATIO), max_width_by_height)
    box_height = int(box_width * OMR_RATIO)
    x1 = max(0, (width - box_width) // 2)
    y1 = max(0, (height - box_height) // 2)
    x2 = min(width, x1 + box_width)
    y2 = min(height, y1 + box_height)
    return x1, y1, x2, y2


def _extract_box_region(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _omr_box_coords(w, h)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _detect_document_quad(image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    min_area = MIN_DOC_AREA_RATIO * image.shape[0] * image.shape[1]
    for contour in contours[:30]:
        if cv2.contourArea(contour) < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return _order_points(approx.reshape(4, 2).astype("float32")), gray, edges

    for contour in contours[:10]:
        if cv2.contourArea(contour) < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return _order_points(box.astype("float32")), gray, edges

    return None, gray, edges


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    dst = np.array(
        [[0, 0], [WARP_WIDTH - 1, 0], [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (WARP_WIDTH, WARP_HEIGHT))


def _enhance_scanned_image(scanned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )
    return gray, thresholded


def _run_tesseract_boxes(gray: np.ndarray) -> list[dict]:
    if shutil.which("tesseract") is None:
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "scan.png"
        out_base = Path(tmpdir) / "ocr"
        cv2.imwrite(str(img_path), gray)
        cmd = [
            "tesseract",
            str(img_path),
            str(out_base),
            "--psm",
            "6",
            "tsv",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return []
        tsv_path = out_base.with_suffix(".tsv")
        if not tsv_path.exists():
            return []

        rows: list[dict] = []
        for line in tsv_path.read_text(errors="ignore").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            text = parts[11].strip()
            if not text:
                continue
            try:
                conf = float(parts[10])
                left = int(parts[6])
                top = int(parts[7])
                width = int(parts[8])
                height = int(parts[9])
            except ValueError:
                continue
            rows.append({
                "text": text.lower(),
                "conf": conf,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            })
        return rows


def _fallback_answer_rect(width: int, height: int) -> tuple[int, int, int, int]:
    x_ratio, y_ratio, w_ratio, h_ratio = ANSWER_RATIO_FALLBACK
    x = int(width * x_ratio)
    y = int(height * y_ratio)
    w = int(width * w_ratio)
    h = int(height * h_ratio)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return x, y, w, h


def _detect_answer_rect(scanned: np.ndarray, thresholded: np.ndarray) -> tuple[int, int, int, int]:
    h, w = scanned.shape[:2]

    # OCR anchor: if "answer" is found, start the crop below it.
    gray = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    ocr_boxes = _run_tesseract_boxes(gray)
    for box in ocr_boxes:
        if "answer" in box["text"] and box["conf"] >= 20:
            top = min(h - 1, box["top"] + box["height"] + int(0.02 * h))
            rect = _fallback_answer_rect(w, h)
            return rect[0], max(top, rect[1]), rect[2], min(h - max(top, rect[1]), rect[3])

    inv = cv2.bitwise_not(thresholded)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int]] = []
    image_area = h * w
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.00003 or area > image_area * 0.015:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if y < int(h * 0.38):
            continue
        if cw < 4 or ch < 4:
            continue
        aspect = cw / max(ch, 1)
        if not 0.35 <= aspect <= 3.0:
            continue
        candidates.append((x, y, cw, ch))

    if len(candidates) >= 8:
        xs = [c[0] for c in candidates]
        ys = [c[1] for c in candidates]
        x2s = [c[0] + c[2] for c in candidates]
        y2s = [c[1] + c[3] for c in candidates]
        pad_x = int(0.03 * w)
        pad_y = int(0.03 * h)
        x = max(0, min(xs) - pad_x)
        y = max(0, min(ys) - pad_y)
        x2 = min(w, max(x2s) + pad_x)
        y2 = min(h, max(y2s) + pad_y)
        if x2 - x > int(w * 0.25) and y2 - y > int(h * 0.10):
            return x, y, x2 - x, y2 - y

    return _fallback_answer_rect(w, h)


def _crop_with_rect(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    ih, iw = image.shape[:2]
    x = max(0, min(iw - 1, int(x)))
    y = max(0, min(ih - 1, int(y)))
    w = max(1, min(iw - x, int(w)))
    h = max(1, min(ih - y, int(h)))
    return image[y : y + h, x : x + w]


def _build_output_from_scanned(scanned: np.ndarray, detected: bool, quad: np.ndarray | None) -> dict:
    gray, thresholded = _enhance_scanned_image(scanned)
    answer_rect = _detect_answer_rect(scanned, thresholded)

    preview = scanned.copy()
    x, y, w, h = answer_rect
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 3)
    final_crop = _crop_with_rect(scanned, answer_rect)

    grayscale_url = _encode_image_data_url(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    threshold_url = _encode_image_data_url(cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR))
    preview_url = _encode_image_data_url(preview)
    cropped_url = _encode_image_data_url(final_crop)
    source_url = _encode_image_data_url(scanned)
    if not grayscale_url or not threshold_url or not preview_url or not cropped_url or not source_url:
        raise ValueError("encode failed")

    return {
        "detected": detected,
        "quad": quad.tolist() if quad is not None else None,
        "captured_image": grayscale_url,
        "pipeline_image": threshold_url,
        "biggest_rect_image": preview_url,
        "cropped_image": cropped_url,
        "crop_source_image": source_url,
        "crop_rect": [x, y, w, h],
        "ocr_enabled": shutil.which("tesseract") is not None,
    }


def _scan_pipeline(image: np.ndarray, quad: np.ndarray | None = None) -> dict:
    roi, (x1, y1, _, _) = _extract_box_region(image)
    detected_quad_roi, _, _ = _detect_document_quad(roi)

    if quad is not None:
        quad_full = quad
        detected = True
    elif detected_quad_roi is not None:
        quad_full = detected_quad_roi.copy()
        quad_full[:, 0] += x1
        quad_full[:, 1] += y1
        detected = True
    else:
        quad_full = _default_quad_for_image(image)
        detected = False

    scanned = four_point_transform(image, quad_full)
    return _build_output_from_scanned(scanned, detected, quad_full if detected else None)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img")
    if not data:
        return jsonify({"detected": False, "stable": False, "can_capture": False})

    image = _decode_data_url_image(data)
    if image is None:
        return jsonify({"detected": False, "stable": False, "can_capture": False})

    roi, _ = _extract_box_region(image)
    quad, _, _ = _detect_document_quad(roi)
    detected = quad is not None
    return jsonify({"detected": detected, "stable": detected, "can_capture": detected})


@app.route("/process", methods=["POST"])
def process():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img")
    if not data:
        return "no image", 400

    image = _decode_data_url_image(data)
    if image is None:
        return "decode failed", 400

    try:
        output = _scan_pipeline(image)
    except ValueError:
        return "encode failed", 400

    output["stable"] = output["detected"]
    return jsonify(output)


@app.route("/prepare", methods=["POST"])
def prepare():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img")
    if not data:
        return "no image", 400

    image = _decode_data_url_image(data)
    if image is None:
        return "decode failed", 400

    roi, (x1, y1, _, _) = _extract_box_region(image)
    quad, _, _ = _detect_document_quad(roi)
    if quad is None:
        quad_full = _default_quad_for_image(image)
    else:
        quad_full = quad.copy()
        quad_full[:, 0] += x1
        quad_full[:, 1] += y1

    return jsonify({"quad_full": quad_full.tolist()})


@app.route("/process_manual", methods=["POST"])
def process_manual():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img")
    points = payload.get("points")
    if not data or not points:
        return "invalid payload", 400

    image = _decode_data_url_image(data)
    if image is None:
        return "decode failed", 400

    try:
        quad = np.array(points, dtype="float32")
    except (TypeError, ValueError):
        return "invalid points", 400

    if quad.shape != (4, 2):
        return "invalid points", 400

    try:
        output = _scan_pipeline(image, quad=quad)
    except ValueError:
        return "encode failed", 400

    output["stable"] = True
    return jsonify(output)


@app.route("/confirm_crop", methods=["POST"])
def confirm_crop():
    payload = request.get_json(silent=True) or {}
    data = payload.get("img")
    rect = payload.get("rect")
    if not data or not rect:
        return "invalid payload", 400

    image = _decode_data_url_image(data)
    if image is None:
        return "decode failed", 400

    try:
        x, y, w, h = [int(v) for v in rect]
    except (TypeError, ValueError):
        return "invalid rect", 400

    final_crop = _crop_with_rect(image, (x, y, w, h))
    cropped_url = _encode_image_data_url(final_crop)
    if not cropped_url:
        return "encode failed", 400

    return jsonify({
        "cropped_image": cropped_url,
        "crop_source_image": data,
        "crop_rect": [x, y, w, h],
        "detected": True,
        "stable": True,
    })


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


# def _manual_crop_with_opencv_gui(image: np.ndarray) -> np.ndarray:
#     # Manual CamScanner-style ROI selection using OpenCV GUI.
#     try:
#         roi = cv2.selectROI("Manual ROI", image, showCrosshair=True, fromCenter=False)
#         cv2.destroyWindow("Manual ROI")
#     except cv2.error:
#         return image

#     x, y, w, h = [int(v) for v in roi]
#     if w <= 0 or h <= 0:
#         return image

#     x2 = min(image.shape[1], x + w)
#     y2 = min(image.shape[0], y + h)
#     return image[max(0, y):y2, max(0, x):x2]


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


# def _default_quad_for_image(image: np.ndarray) -> np.ndarray:
#     h, w = image.shape[:2]
#     return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")


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


# def _process_from_manual_quad(roi: np.ndarray, quad: np.ndarray) -> dict:
#     warped = four_point_transform(roi, quad)
#     captured_doc, doc_edges, doc_quad = _crop_biggest_rectangle(warped)
#     captured_doc = cv2.resize(captured_doc, (WARP_WIDTH, WARP_HEIGHT), interpolation=cv2.INTER_LINEAR)
#     processed = apply_lighting_normalization(captured_doc)

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

#     pre_crop_vis = captured_doc.copy()
#     cv2.rectangle(pre_crop_vis, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 3)
#     pipeline_preview = _make_pipeline_preview(captured_doc, doc_edges, doc_quad, pre_crop_vis)

#     captured_url = _encode_image_data_url(roi, quality=92)
#     pipeline_url = _encode_image_data_url(pipeline_preview, quality=92)
#     biggest_rect_url = _encode_image_data_url(captured_doc, quality=92)
#     cropped_url = _encode_image_data_url(answer, quality=92)

#     if not captured_url or not pipeline_url or not biggest_rect_url or not cropped_url:
#         raise ValueError("encode failed")

#     return {
#         "captured_image": captured_url,
#         "pipeline_image": pipeline_url,
#         "biggest_rect_image": biggest_rect_url,
#         "cropped_image": cropped_url,
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

#     # Optional manual crop step before existing OMR pipeline.
#     if bool(payload.get("manual_gui")):
#         img = _manual_crop_with_opencv_gui(img)

#     frame_info = process_frame(img)
#     try:
#         output = _process_from_manual_quad(frame_info["roi"], frame_info["quad"] if frame_info["quad"] is not None else _default_quad_for_image(frame_info["roi"]))
#     except ValueError:
#         return "encode failed", 400

#     output["stable"] = frame_info["stable"]
#     return jsonify(output)


# @app.route("/prepare", methods=["POST"])
# def prepare():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("img")
#     if not data:
#         return "no image", 400

#     img = _decode_data_url_image(data)
#     if img is None:
#         return "decode failed", 400

#     roi = _extract_box_region(img)
#     quad, _, _ = detect_document(roi)
#     if quad is None:
#         quad = _default_quad_for_image(roi)

#     h, w = img.shape[:2]
#     x1, y1, _, _ = _omr_box_coords(w, h)
#     quad_full = quad.copy()
#     quad_full[:, 0] += x1
#     quad_full[:, 1] += y1

#     roi_url = _encode_image_data_url(roi, quality=92)
#     if not roi_url:
#         return "encode failed", 400

#     return jsonify({"editor_image": roi_url, "quad": quad.tolist(), "quad_full": quad_full.tolist()})


# @app.route("/process_manual", methods=["POST"])
# def process_manual():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("img")
#     points = payload.get("points")
#     if not data or not points:
#         return "invalid payload", 400

#     roi = _decode_data_url_image(data)
#     if roi is None:
#         return "decode failed", 400

#     try:
#         quad = np.array(points, dtype="float32")
#     except (TypeError, ValueError):
#         return "invalid points", 400

#     if quad.shape != (4, 2):
#         return "invalid points", 400

#     try:
#         output = _process_from_manual_quad(roi, quad)
#     except ValueError:
#         return "encode failed", 400

#     output["stable"] = True
#     return jsonify(output)


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=10000)





