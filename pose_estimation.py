import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from calibration_data import CAMERA_MATRIX as DEFAULT_CAMERA_MATRIX
    from calibration_data import DIST_COEFFS as DEFAULT_DIST_COEFFS
    from calibration_data import PATTERN_SIZE as DEFAULT_PATTERN_SIZE
    from calibration_data import SQUARE_SIZE as DEFAULT_SQUARE_SIZE
    from calibration_data import IMAGE_SIZE as DEFAULT_IMAGE_SIZE
except ImportError:
    DEFAULT_CAMERA_MATRIX = None
    DEFAULT_DIST_COEFFS = None
    DEFAULT_PATTERN_SIZE = None
    DEFAULT_SQUARE_SIZE = 1.0
    DEFAULT_IMAGE_SIZE = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"}


class PoseEstimationError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate chessboard pose from calibrated camera parameters and render a custom AR object."
    )
    parser.add_argument(
        "--calibration-mode",
        choices=("python", "file", "runtime"),
        default="python",
        help="Calibration source: python constants, .npz/.json file, or runtime calibration from chessboard images/video.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Path to calibration result .npz or .json file. Used only with --calibration-mode file.",
    )
    parser.add_argument(
        "--calibration-images",
        default=None,
        help="Image directory or video used for --calibration-mode runtime. Defaults to --input.",
    )
    parser.add_argument(
        "--min-calibration-frames",
        type=int,
        default=8,
        help="Minimum valid chessboard detections required for runtime calibration.",
    )
    parser.add_argument(
        "--input",
        default="data/captures",
        help="Input image, image directory, or video file. Ignored when --camera is provided.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index for live pose estimation, for example 0.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=960,
        help="Requested camera width for live preview.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=540,
        help="Requested camera height for live preview.",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=30.0,
        help="Requested camera FPS for live preview.",
    )
    parser.add_argument(
        "--pattern-cols",
        type=int,
        default=10,
        help="Number of chessboard inner corners along the width.",
    )
    parser.add_argument(
        "--pattern-rows",
        type=int,
        default=7,
        help="Number of chessboard inner corners along the height.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=None,
        help="Real-world square size. Defaults to the Python calibration constant or calibration file value.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use one frame every N frames when reading a video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames/images to process. Use 0 for no limit.",
    )
    parser.add_argument(
        "--output",
        default="outputs/pose_demo.png",
        help="Demo image path. For a folder input, the first successful pose frame is saved here.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pose_frames",
        help="Directory for per-image pose visualizations.",
    )
    parser.add_argument(
        "--video-output",
        default="outputs/pose_demo.avi",
        help="Output video path for video or camera input.",
    )
    parser.add_argument(
        "--results-json",
        default="outputs/pose_results.json",
        help="Path to save pose vectors and detection metadata.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=3.0,
        help="3D axis length in square-size units.",
    )
    parser.add_argument(
        "--detection-scale",
        type=float,
        default=0.5,
        help="Scale used for chessboard detection. Lower values improve preview FPS.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scale used only for the preview window.",
    )
    parser.add_argument(
        "--no-video-save",
        action="store_true",
        help="Skip VideoWriter output for faster live preview.",
    )
    parser.add_argument(
        "--record-on-start",
        action="store_true",
        help="Start video recording immediately instead of waiting for the R key.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Text to render as an AR object. If omitted, the program asks for input at startup.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable preview window for video/camera input.",
    )
    return parser.parse_args()


def load_calibration(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        camera_matrix = data["camera_matrix"] if "camera_matrix" in data.files else data["K"]
        dist_coeffs = data["dist_coeffs"] if "dist_coeffs" in data.files else data["dist_coeff"]
        square_size = float(np.ravel(data["square_size"])[0]) if "square_size" in data.files else 1.0
        pattern = None
        if "requested_pattern" in data.files:
            requested = np.ravel(data["requested_pattern"]).astype(int)
            if requested.size >= 2:
                pattern = (int(requested[0]), int(requested[1]))
        image_size = None
        if "image_size" in data.files:
            image_size_value = np.ravel(data["image_size"]).astype(int)
            if image_size_value.size >= 2:
                image_size = (int(image_size_value[0]), int(image_size_value[1]))
        return np.asarray(camera_matrix, dtype=np.float64), np.asarray(dist_coeffs, dtype=np.float64), square_size, pattern, image_size

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        camera_matrix = data.get("camera_matrix") or data.get("K")
        dist_coeffs = data.get("dist_coeffs") or data.get("dist_coeff")
        if camera_matrix is None or dist_coeffs is None:
            raise PoseEstimationError("JSON calibration must contain camera_matrix/K and dist_coeffs/dist_coeff.")
        square_size = float(data.get("square_size", 1.0))
        pattern_value = data.get("requested_pattern")
        pattern = tuple(pattern_value[:2]) if pattern_value else None
        image_size_value = data.get("image_size")
        image_size = tuple(image_size_value[:2]) if image_size_value else None
        return np.asarray(camera_matrix, dtype=np.float64), np.asarray(dist_coeffs, dtype=np.float64), square_size, pattern, image_size

    raise PoseEstimationError("Calibration file must be .npz or .json.")


def load_python_calibration():
    if DEFAULT_CAMERA_MATRIX is None or DEFAULT_DIST_COEFFS is None:
        raise PoseEstimationError("calibration_data.py is missing default camera constants.")
    pattern = tuple(DEFAULT_PATTERN_SIZE) if DEFAULT_PATTERN_SIZE is not None else None
    return (
        np.asarray(DEFAULT_CAMERA_MATRIX, dtype=np.float64),
        np.asarray(DEFAULT_DIST_COEFFS, dtype=np.float64),
        float(DEFAULT_SQUARE_SIZE),
        pattern,
        tuple(DEFAULT_IMAGE_SIZE) if DEFAULT_IMAGE_SIZE is not None else None,
    )


def resolve_ar_text(value):
    if value is not None:
        text_value = str(value).strip()
    else:
        try:
            text_value = input("AR text (default AR): ").strip()
        except EOFError:
            text_value = ""
    return text_value or "AR"


def candidate_patterns(pattern_size):
    swapped = (pattern_size[1], pattern_size[0])
    return [pattern_size] if swapped == pattern_size else [pattern_size, swapped]


def make_object_points(pattern_size, square_size):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def refine_corners(gray, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def preprocess_grayscale(gray):
    candidates = [("gray", gray)]
    candidates.append(("equalized", cv2.equalizeHist(gray)))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    candidates.append(("clahe", clahe.apply(gray)))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    candidates.append(("blurred", blurred))
    return candidates


def try_find_corners_sb(gray_for_detection, pattern_size):
    if not hasattr(cv2, "findChessboardCornersSB"):
        return False, None, None

    flags = 0
    for attr_name in ("CALIB_CB_NORMALIZE_IMAGE", "CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY"):
        if hasattr(cv2, attr_name):
            flags |= getattr(cv2, attr_name)

    found, corners = cv2.findChessboardCornersSB(gray_for_detection, pattern_size, flags)
    if not found:
        return False, None, None
    return True, corners.astype(np.float32), "findChessboardCornersSB"


def try_find_corners_classic(gray_for_detection, gray_for_refine, pattern_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray_for_detection, pattern_size, flags)
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray_for_detection, pattern_size, flags)
    if not found:
        return False, None, None
    return True, refine_corners(gray_for_refine, corners), "findChessboardCorners"


def detect_chessboard(frame, pattern_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates = preprocess_grayscale(gray)

    for active_pattern in candidate_patterns(pattern_size):
        for candidate_name, candidate in candidates:
            found, corners, method = try_find_corners_sb(candidate, active_pattern)
            if found:
                return True, corners, gray, active_pattern, f"{method}:{candidate_name}"

        for candidate_name, candidate in candidates:
            found, corners, method = try_find_corners_classic(candidate, gray, active_pattern)
            if found:
                return True, corners, gray, active_pattern, f"{method}:{candidate_name}"

    return False, None, gray, None, None


def detect_chessboard_fast(frame, pattern_size, detection_scale):
    scale = float(detection_scale)
    if scale <= 0 or scale > 1.0:
        scale = 1.0
    if scale >= 0.999:
        return detect_chessboard(frame, pattern_size)

    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    found, corners, gray, active_pattern, method = detect_chessboard(small, pattern_size)
    if found:
        corners = (corners / scale).astype(np.float32)
        method = f"{method}:scale={scale:.2f}"
    return found, corners, gray, active_pattern, method


def scale_camera_matrix(camera_matrix, calibration_image_size, frame_size):
    if calibration_image_size is None:
        return camera_matrix
    calib_w, calib_h = calibration_image_size
    frame_w, frame_h = frame_size
    if calib_w <= 0 or calib_h <= 0 or (calib_w, calib_h) == (frame_w, frame_h):
        return camera_matrix
    scaled = camera_matrix.copy()
    scaled[0, 0] *= frame_w / calib_w
    scaled[0, 2] *= frame_w / calib_w
    scaled[1, 1] *= frame_h / calib_h
    scaled[1, 2] *= frame_h / calib_h
    return scaled


def project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs):
    points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    projected, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    return np.round(projected.reshape(-1, 2)).astype(int)


def draw_line_3d(frame, rvec, tvec, camera_matrix, dist_coeffs, p1, p2, color, thickness=3):
    pts = project_points([p1, p2], rvec, tvec, camera_matrix, dist_coeffs)
    cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, thickness, cv2.LINE_AA)


def draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, axis_length):
    origin = (0.0, 0.0, 0.0)
    draw_line_3d(frame, rvec, tvec, camera_matrix, dist_coeffs, origin, (axis_length, 0.0, 0.0), (0, 0, 255), 4)
    draw_line_3d(frame, rvec, tvec, camera_matrix, dist_coeffs, origin, (0.0, axis_length, 0.0), (0, 255, 0), 4)
    draw_line_3d(frame, rvec, tvec, camera_matrix, dist_coeffs, origin, (0.0, 0.0, -axis_length), (255, 0, 0), 4)

    label_points = project_points(
        [(axis_length, 0.0, 0.0), (0.0, axis_length, 0.0), (0.0, 0.0, -axis_length)],
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    for label, point, color in zip(("X", "Y", "Z"), label_points, ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.putText(frame, label, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def draw_pyramid(frame, rvec, tvec, camera_matrix, dist_coeffs, pattern_size, square_size):
    cols, rows = pattern_size
    board_w = (cols - 1) * square_size
    board_h = (rows - 1) * square_size
    center_x = board_w * 0.5
    center_y = board_h * 0.5
    base_w = min(board_w, board_h) * 0.45
    base_h = min(board_w, board_h) * 0.38
    height = min(board_w, board_h) * 0.55

    points_3d = np.asarray(
        [
            (center_x - base_w, center_y - base_h, 0.0),
            (center_x + base_w, center_y - base_h, 0.0),
            (center_x + base_w, center_y + base_h, 0.0),
            (center_x - base_w, center_y + base_h, 0.0),
            (center_x, center_y, -height),
        ],
        dtype=np.float32,
    )
    pts = project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs)

    overlay = frame.copy()
    faces = [
        ([pts[0], pts[1], pts[4]], (30, 220, 255)),
        ([pts[1], pts[2], pts[4]], (40, 170, 255)),
        ([pts[2], pts[3], pts[4]], (50, 120, 255)),
        ([pts[3], pts[0], pts[4]], (60, 80, 255)),
    ]
    for face_points, color in faces:
        cv2.fillConvexPoly(overlay, np.asarray(face_points, dtype=np.int32), color, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.32, frame, 0.68, 0.0, frame)

    base_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edge_indices = base_indices + [(0, 4), (1, 4), (2, 4), (3, 4)]
    for a, b in edge_indices:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 60, 255), 3, cv2.LINE_AA)


def make_text_rgba(text, width=960, height=260):
    display_text = str(text or "AR").strip() or "AR"
    if len(display_text) > 24:
        display_text = display_text[:24]

    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 10
    scale = 4.0
    max_width = int(width * 0.88)
    max_height = int(height * 0.62)

    while scale > 0.35:
        (text_w, text_h), baseline = cv2.getTextSize(display_text, font, scale, thickness)
        if text_w <= max_width and text_h + baseline <= max_height:
            break
        scale *= 0.9
        thickness = max(2, int(thickness * 0.92))

    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, scale, thickness)
    x = max(8, (width - text_w) // 2)
    y = max(text_h + 8, (height + text_h) // 2 - baseline)

    cv2.putText(canvas, display_text, (x + 7, y + 7), font, scale, (0, 0, 0, 150), thickness + 5, cv2.LINE_AA)
    cv2.putText(canvas, display_text, (x, y), font, scale, (255, 255, 255, 255), thickness + 4, cv2.LINE_AA)
    cv2.putText(canvas, display_text, (x, y), font, scale, (255, 230, 20, 255), thickness, cv2.LINE_AA)
    cv2.rectangle(canvas, (18, 18), (width - 18, height - 18), (255, 230, 20, 210), 6, cv2.LINE_AA)
    return canvas


def blend_rgba_warp(frame, rgba, dst_pts):
    h, w = rgba.shape[:2]
    src_pts = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    transform = cv2.getPerspectiveTransform(src_pts, np.asarray(dst_pts, dtype=np.float32))
    warped = cv2.warpPerspective(
        rgba,
        transform,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    color = warped[:, :, :3].astype(np.float32)
    base = frame.astype(np.float32)
    frame[:] = np.clip(color * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)


def draw_ar_text(frame, rvec, tvec, camera_matrix, dist_coeffs, pattern_size, square_size, text):
    cols, rows = pattern_size
    board_w = (cols - 1) * square_size
    board_h = (rows - 1) * square_size
    plane_w = board_w * 0.72
    plane_h = board_h * 0.22
    x0 = board_w * 0.5 - plane_w * 0.5
    x1 = x0 + plane_w
    y0 = board_h * 0.11
    y1 = y0 + plane_h
    z = -min(board_w, board_h) * 0.34

    model_pts = np.float32([
        [x0, y0, z],
        [x0, y1, z],
        [x1, y0, z],
        [x1, y1, z],
    ])
    dst_pts = project_points(model_pts, rvec, tvec, camera_matrix, dist_coeffs)
    rgba = make_text_rgba(text)
    blend_rgba_warp(frame, rgba, dst_pts)

    for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):
        cv2.line(frame, tuple(dst_pts[a]), tuple(dst_pts[b]), (255, 255, 40), 2, cv2.LINE_AA)


def camera_position_from_pose(rvec, tvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_position = -rotation_matrix.T @ tvec.reshape(3, 1)
    return camera_position.reshape(-1)


def annotate_status(frame, source_label, detected, method=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 66), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    status = "POSE OK" if detected else "CHESSBOARD NOT DETECTED"
    color = (70, 255, 110) if detected else (40, 70, 255)
    cv2.putText(frame, status, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    cv2.putText(frame, str(source_label), (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    if method:
        cv2.putText(frame, method, (frame.shape[1] - 430, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)


def estimate_and_render(frame, source_label, camera_matrix, dist_coeffs, pattern_size, square_size, axis_length, ar_text, detection_scale=1.0):
    annotated = frame.copy()
    found, corners, _gray, active_pattern, method = detect_chessboard_fast(frame, pattern_size, detection_scale)

    result = {
        "source": str(source_label),
        "detected": bool(found),
        "requested_pattern": f"{pattern_size[0]}x{pattern_size[1]}",
        "used_pattern": None,
        "method": method,
        "rvec": None,
        "tvec": None,
        "camera_position_board": None,
    }

    if not found:
        annotate_status(annotated, source_label, False)
        return annotated, result

    object_points = make_object_points(active_pattern, square_size)
    ok, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        annotate_status(annotated, source_label, False, "solvePnP failed")
        return annotated, result

    cv2.drawChessboardCorners(annotated, active_pattern, corners, True)
    draw_axes(annotated, rvec, tvec, camera_matrix, dist_coeffs, axis_length * square_size)
    draw_pyramid(annotated, rvec, tvec, camera_matrix, dist_coeffs, active_pattern, square_size)
    draw_ar_text(annotated, rvec, tvec, camera_matrix, dist_coeffs, active_pattern, square_size, ar_text)
    annotate_status(annotated, source_label, True, method)

    result.update(
        {
            "used_pattern": f"{active_pattern[0]}x{active_pattern[1]}",
            "rvec": [float(x) for x in rvec.reshape(-1)],
            "tvec": [float(x) for x in tvec.reshape(-1)],
            "camera_position_board": [float(x) for x in camera_position_from_pose(rvec, tvec)],
        }
    )
    return annotated, result


def list_images(input_dir):
    input_dir = Path(input_dir)
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def iter_calibration_frames(source, frame_step, max_frames):
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Calibration source not found: {source_path}")

    if source_path.is_dir():
        emitted = 0
        for image_path in list_images(source_path):
            if max_frames > 0 and emitted >= max_frames:
                break
            frame = cv2.imread(str(image_path))
            if frame is not None:
                yield image_path.name, frame
                emitted += 1
        return

    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(source_path))
        if frame is not None:
            yield source_path.name, frame
        return

    if source_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise PoseEstimationError("Runtime calibration source must be an image, image directory, or video file.")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise PoseEstimationError(f"Cannot open calibration video: {source_path}")

    frame_index = 0
    emitted = 0
    try:
        while max_frames <= 0 or emitted < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % max(1, frame_step) == 0:
                yield f"frame_{frame_index:05d}", frame
                emitted += 1
            frame_index += 1
    finally:
        cap.release()


def calibrate_camera_from_source(source, pattern_size, square_size, frame_step, max_frames, min_detections):
    objpoints = []
    imgpoints = []
    image_size = None
    active_pattern = None
    used_sources = []

    for label, frame in iter_calibration_frames(source, frame_step, max_frames):
        found, corners, gray, detected_pattern, method = detect_chessboard(frame, pattern_size)
        if not found:
            continue
        if active_pattern is None:
            active_pattern = detected_pattern
        if detected_pattern != active_pattern:
            continue
        objpoints.append(make_object_points(active_pattern, square_size))
        imgpoints.append(corners)
        image_size = gray.shape[::-1]
        used_sources.append({"source": label, "method": method})

    if len(imgpoints) < min_detections:
        raise PoseEstimationError(
            f"Runtime calibration needs at least {min_detections} detections, but found {len(imgpoints)}."
        )

    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    metadata = {
        "source": str(source),
        "mode": "runtime",
        "rms": float(rms),
        "image_size": list(image_size),
        "used_sources": used_sources,
    }
    return camera_matrix, dist_coeffs, square_size, active_pattern, metadata, image_size


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_image_file(image_path, args, camera_matrix, dist_coeffs, pattern_size, square_size):
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise PoseEstimationError(f"Cannot read image: {image_path}")
    frame_camera_matrix = scale_camera_matrix(camera_matrix, args.calibration_image_size, (frame.shape[1], frame.shape[0]))
    annotated, result = estimate_and_render(
        frame,
        image_path.name,
        frame_camera_matrix,
        dist_coeffs,
        pattern_size,
        square_size,
        args.axis_length,
        args.ar_text,
        args.detection_scale,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    result["output_file"] = str(output_path)
    return [result]


def process_image_directory(input_dir, args, camera_matrix, dist_coeffs, pattern_size, square_size):
    image_paths = list_images(input_dir)
    if not image_paths:
        raise PoseEstimationError(f"No images found in: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_output = Path(args.output)
    demo_output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    demo_saved = False
    processed = 0
    for image_path in image_paths:
        if args.max_frames > 0 and processed >= args.max_frames:
            break
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        frame_camera_matrix = scale_camera_matrix(camera_matrix, args.calibration_image_size, (frame.shape[1], frame.shape[0]))
        annotated, result = estimate_and_render(
            frame,
            image_path.name,
            frame_camera_matrix,
            dist_coeffs,
            pattern_size,
            square_size,
            args.axis_length,
            args.ar_text,
            args.detection_scale,
        )
        output_path = output_dir / f"{image_path.stem}_pose.png"
        cv2.imwrite(str(output_path), annotated)
        result["output_file"] = str(output_path)
        results.append(result)
        if result["detected"] and not demo_saved:
            cv2.imwrite(str(demo_output), annotated)
            demo_saved = True
        processed += 1

    if not demo_saved and results:
        fallback_path = Path(results[0]["output_file"])
        fallback = cv2.imread(str(fallback_path))
        if fallback is not None:
            cv2.imwrite(str(demo_output), fallback)

    return results


def make_video_writer(output_path, frame_size, fps):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    fourcc_code = "mp4v" if suffix == ".mp4" else "MJPG"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc_code), fps, frame_size)
    if not writer.isOpened():
        raise PoseEstimationError(f"Cannot open video writer: {output_path}")
    return writer


def draw_recording_badge(frame, recording, can_record):
    if not can_record:
        text = "PREVIEW ONLY"
        color = (180, 180, 180)
    elif recording:
        text = "REC  R:pause  ESC:quit"
        color = (0, 0, 255)
    else:
        text = "PREVIEW  R:record  ESC:quit"
        color = (80, 220, 255)

    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x = max(12, frame.shape[1] - text_w - 24)
    y = frame.shape[0] - 24
    cv2.rectangle(frame, (x - 10, y - text_h - 10), (x + text_w + 10, y + baseline + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def process_video_source(source, args, camera_matrix, dist_coeffs, pattern_size, square_size):
    is_camera = isinstance(source, int)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise PoseEstimationError(f"Cannot open video source: {source}")

    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = args.camera_fps if is_camera else 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        ok, sample = cap.read()
        if not ok:
            cap.release()
            raise PoseEstimationError(f"Cannot read video source: {source}")
        height, width = sample.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = None
    can_record = not args.no_video_save
    recording = can_record and (args.no_window or args.record_on_start)

    print(
        f"[INFO] Video source: {source} | {width}x{height} @ {fps:.1f} fps | "
        f"detection_scale={args.detection_scale} | display_scale={args.display_scale}"
    )
    if args.no_video_save:
        print("[INFO] VideoWriter disabled for faster preview.")
    elif args.no_window:
        print(f"[INFO] No-window mode: recording automatically to {args.video_output}")
    elif args.record_on_start:
        print(f"[INFO] Recording starts immediately. Press R to pause/resume, ESC to quit. Output: {args.video_output}")
    else:
        print("[INFO] Preview mode. Press R to start/pause recording, ESC to quit.")

    results = []
    frame_index = 0
    emitted = 0
    try:
        while args.max_frames <= 0 or emitted < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % max(1, args.frame_step) != 0:
                frame_index += 1
                continue

            label = f"frame_{frame_index:05d}"
            frame_camera_matrix = scale_camera_matrix(
                camera_matrix,
                args.calibration_image_size,
                (frame.shape[1], frame.shape[0]),
            )
            annotated, result = estimate_and_render(
                frame,
                label,
                frame_camera_matrix,
                dist_coeffs,
                pattern_size,
                square_size,
                args.axis_length,
                args.ar_text,
                args.detection_scale,
            )
            draw_recording_badge(annotated, recording, can_record)

            if recording and can_record:
                if writer is None:
                    writer = make_video_writer(args.video_output, (frame.shape[1], frame.shape[0]), fps)
                    print(f"[INFO] Recording started: {args.video_output}")
                writer.write(annotated)
                result["output_file"] = str(args.video_output)
            else:
                result["output_file"] = None
            result["recording"] = bool(recording and can_record)
            results.append(result)
            emitted += 1

            if not args.no_window:
                preview = annotated
                display_scale = float(args.display_scale)
                if 0 < display_scale < 0.999:
                    preview = cv2.resize(annotated, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
                cv2.imshow("BoardPoseAR", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key in (ord("r"), ord("R")):
                    if can_record:
                        recording = not recording
                        print("[INFO] Recording " + ("ON" if recording else "PAUSED"))
                    else:
                        print("[INFO] --no-video-save is active; recording is disabled.")
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    return results


def build_summary(results, calibration_metadata, pattern_size, square_size, ar_text):
    detected = sum(1 for item in results if item.get("detected"))
    attempted = len(results)
    return {
        "calibration": calibration_metadata,
        "pattern": f"{pattern_size[0]}x{pattern_size[1]}",
        "square_size": float(square_size),
        "ar_text": ar_text,
        "attempted_frames_or_images": attempted,
        "detected_poses": detected,
        "results": results,
    }


def resolve_calibration(args):
    requested_pattern = (args.pattern_cols, args.pattern_rows)
    requested_square_size = args.square_size if args.square_size is not None else float(DEFAULT_SQUARE_SIZE)

    mode = args.calibration_mode
    if mode == "python" and args.calibration is not None:
        mode = "file"

    if mode == "python":
        camera_matrix, dist_coeffs, calib_square_size, calib_pattern, calib_image_size = load_python_calibration()
        metadata = {"mode": "python", "source": "calibration_data.py"}
    elif mode == "file":
        if not args.calibration:
            raise PoseEstimationError("--calibration-mode file requires --calibration PATH.")
        calibration_path = args.calibration
        camera_matrix, dist_coeffs, calib_square_size, calib_pattern, calib_image_size = load_calibration(calibration_path)
        metadata = {"mode": "file", "source": str(calibration_path)}
    else:
        if args.camera is not None and args.calibration_images is None:
            raise PoseEstimationError("--calibration-mode runtime with --camera requires --calibration-images.")
        calibration_source = args.calibration_images or args.input
        camera_matrix, dist_coeffs, calib_square_size, calib_pattern, metadata, calib_image_size = calibrate_camera_from_source(
            calibration_source,
            requested_pattern,
            requested_square_size,
            args.frame_step,
            args.max_frames,
            args.min_calibration_frames,
        )

    pattern_size = requested_pattern
    if calib_pattern is not None and requested_pattern == (10, 7):
        pattern_size = tuple(int(x) for x in calib_pattern)
    square_size = args.square_size if args.square_size is not None else calib_square_size
    return camera_matrix, dist_coeffs, pattern_size, square_size, metadata, calib_image_size


def main():
    args = parse_args()
    args.ar_text = resolve_ar_text(args.text)
    camera_matrix, dist_coeffs, pattern_size, square_size, calibration_metadata, calibration_image_size = resolve_calibration(args)
    args.calibration_image_size = calibration_image_size

    if args.camera is not None:
        results = process_video_source(args.camera, args, camera_matrix, dist_coeffs, pattern_size, square_size)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        if input_path.is_dir():
            results = process_image_directory(input_path, args, camera_matrix, dist_coeffs, pattern_size, square_size)
        elif input_path.suffix.lower() in IMAGE_EXTENSIONS:
            results = process_image_file(input_path, args, camera_matrix, dist_coeffs, pattern_size, square_size)
        elif input_path.suffix.lower() in VIDEO_EXTENSIONS:
            results = process_video_source(str(input_path), args, camera_matrix, dist_coeffs, pattern_size, square_size)
        else:
            raise PoseEstimationError("Input must be an image, image directory, video, or --camera index.")

    summary = build_summary(results, calibration_metadata, pattern_size, square_size, args.ar_text)
    save_json(args.results_json, summary)

    print(f"[INFO] Calibration mode: {calibration_metadata.get('mode')}")
    print(f"[INFO] Pattern: {pattern_size[0]}x{pattern_size[1]}, square_size={square_size}")
    print(f"[INFO] AR text: {args.ar_text}")
    print(f"[INFO] Pose detections: {summary['detected_poses']} / {summary['attempted_frames_or_images']}")
    print(f"[INFO] Results JSON: {args.results_json}")
    if args.camera is None and Path(args.input).suffix.lower() not in VIDEO_EXTENSIONS:
        print(f"[INFO] Demo image: {args.output}")
    else:
        print(f"[INFO] Demo video: {args.video_output}")


if __name__ == "__main__":
    main()
