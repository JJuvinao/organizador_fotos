import numpy as np
from pathlib import Path
from PIL import Image as PILImage

from mediapipe.tasks.python.vision import (
    FaceDetector as MPFaceDetector,
    FaceDetectorOptions,
    RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp',
    '.tiff', '.tif', '.avif', '.heic', '.heif',
    '.jfif', '.pjpeg', '.pjp'
}

MODEL_PATH = Path(__file__).parent / 'face_detection.tflite'

_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.3,
)
_detector = MPFaceDetector.create_from_options(_options)


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def scan_images(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"La carpeta {folder_path} no existe.")

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.rglob(f'*{ext}'))
        images.extend(folder.rglob(f'*{ext.upper()}'))

    images = sorted(set(images))
    return [str(p) for p in images]


def _load_image_rgb(image_path):
    try:
        pil = PILImage.open(image_path).convert('RGB')
        return np.array(pil), None
    except Exception as e:
        return None, str(e)


def detect_faces_in_image(image_path):
    rgb, error = _load_image_rgb(image_path)
    if rgb is None:
        return [], error

    mp_image = _mediapipe_image_from_array(rgb)
    detections = _detector.detect(mp_image)

    if not detections.detections:
        return [], None

    bgr = rgb[:, :, ::-1]

    faces = []
    h, w, _ = bgr.shape
    MARGIN_FACTOR = 0.2

    raw_detections = []
    for detection in detections.detections:
        bbox = detection.bounding_box
        x = bbox.origin_x
        y = bbox.origin_y
        bw = bbox.width
        bh = bbox.height

        mx = int(bw * MARGIN_FACTOR)
        my = int(bh * MARGIN_FACTOR)
        x = max(0, x - mx)
        y = max(0, y - my)
        bw = min(w - x, bw + 2 * mx)
        bh = min(h - y, bh + 2 * my)

        face_crop = bgr[y:y + bh, x:x + bw]
        if face_crop.size == 0:
            continue

        raw_detections.append({
            "bbox": (x, y, bw, bh),
            "crop": face_crop
        })

    keep = []
    for i, a in enumerate(raw_detections):
        x1, y1, w1, h1 = a["bbox"]
        is_duplicate = False
        for j in keep:
            x2, y2, w2, h2 = raw_detections[j]["bbox"]
            xi = max(x1, x2)
            yi = max(y1, y2)
            wi = min(x1 + w1, x2 + w2) - xi
            hi = min(y1 + h1, y2 + h2) - yi
            if wi > 0 and hi > 0:
                inter = wi * hi
                union = w1 * h1 + w2 * h2 - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.5:
                    is_duplicate = True
                    break
        if not is_duplicate:
            keep.append(i)

    faces = [raw_detections[i] for i in keep]
    return faces, None


def _mediapipe_image_from_array(rgb_array):
    from mediapipe import Image as MPImage, ImageFormat
    return MPImage(ImageFormat.SRGB, rgb_array)


def process_folder(folder_path):
    image_paths = scan_images(folder_path)
    results = []
    no_face_images = []
    errors = []

    for img_path in image_paths:
        try:
            faces, error = detect_faces_in_image(img_path)
            if error:
                errors.append(error)
                continue
            if not faces:
                no_face_images.append(img_path)
            else:
                results.append({
                    'path': img_path,
                    'faces': faces
                })
        except Exception as e:
            errors.append(f"{img_path}: {e}")
            continue

    return results, no_face_images, errors
