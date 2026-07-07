import numpy as np
from PIL import Image
import cv2

FACE_MATCH_THRESHOLD = 0.63
CROP_SIZE = (128, 128)

HOG_CELL_SIZE = 8
HOG_BLOCK_SIZE = 2
HOG_BLOCK_STRIDE = 1
HOG_NBINS = 9


def _bgr_to_gray(bgr):
    return np.dot(bgr[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)


def _resize(gray, size):
    pil = Image.fromarray(gray)
    resized = pil.resize(size, Image.LANCZOS)
    return np.array(resized)


def _preprocess_face(face_crop):
    gray = _bgr_to_gray(face_crop)
    resized = _resize(gray, CROP_SIZE)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0.5)
    return blurred


def _compute_hog_features(gray_img):
    h, w = gray_img.shape
    num_cells_y = h // HOG_CELL_SIZE
    num_cells_x = w // HOG_CELL_SIZE

    if num_cells_y < 2 or num_cells_x < 2:
        return np.array([])

    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx) * 180.0 / np.pi
    ang[ang < 0] += 180.0

    cell_hist = np.zeros((num_cells_y, num_cells_x, HOG_NBINS), dtype=np.float32)
    bin_width = 180.0 / HOG_NBINS
    for cy in range(num_cells_y):
        for cx in range(num_cells_x):
            y0, y1 = cy * HOG_CELL_SIZE, (cy + 1) * HOG_CELL_SIZE
            x0, x1 = cx * HOG_CELL_SIZE, (cx + 1) * HOG_CELL_SIZE
            cell_mag = mag[y0:y1, x0:x1]
            cell_ang = ang[y0:y1, x0:x1]
            for by in range(cell_mag.shape[0]):
                for bx in range(cell_mag.shape[1]):
                    bin_idx = min(int(cell_ang[by, bx] / bin_width), HOG_NBINS - 1)
                    cell_hist[cy, cx, bin_idx] += cell_mag[by, bx]

    features = []
    for cy in range(0, num_cells_y - HOG_BLOCK_SIZE + 1, HOG_BLOCK_STRIDE):
        for cx in range(0, num_cells_x - HOG_BLOCK_SIZE + 1, HOG_BLOCK_STRIDE):
            block = cell_hist[cy:cy + HOG_BLOCK_SIZE, cx:cx + HOG_BLOCK_SIZE, :].flatten()
            norm = np.linalg.norm(block)
            if norm > 1e-6:
                block = block / norm
            block = np.clip(block, 0, 0.2)
            norm = np.linalg.norm(block)
            if norm > 1e-6:
                block = block / norm
            features.append(block)

    if not features:
        return np.array([])

    return np.concatenate(features).astype(np.float32)


def _compute_histogram_similarity(img_a, img_b):
    hist_a, _ = np.histogram(img_a, bins=64, range=(0, 256))
    hist_b, _ = np.histogram(img_b, bins=64, range=(0, 256))
    hist_a = hist_a.astype(float)
    hist_b = hist_b.astype(float)
    if hist_a.sum() > 0:
        hist_a = hist_a / hist_a.sum()
    if hist_b.sum() > 0:
        hist_b = hist_b / hist_b.sum()
    correlation = np.corrcoef(hist_a, hist_b)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    return max(0, correlation)


def _compute_pixel_similarity(img_a, img_b):
    flat_a = img_a.flatten().astype(float)
    flat_b = img_b.flatten().astype(float)
    na = np.linalg.norm(flat_a)
    nb = np.linalg.norm(flat_b)
    if na == 0 or nb == 0:
        return 0.0
    dot = float(np.dot(flat_a / na, flat_b / nb))
    return max(0, min(1, (dot + 1) / 2))


def _compute_similarity(processed_a, processed_b, hog_a, hog_b):
    if hog_a.size > 0 and hog_b.size > 0:
        na = np.linalg.norm(hog_a)
        nb = np.linalg.norm(hog_b)
        if na > 0 and nb > 0:
            return float(np.dot(hog_a / na, hog_b / nb))
    return 0.0


def faces_match(face_crop_a, face_crop_b):
    processed_a = _preprocess_face(face_crop_a)
    processed_b = _preprocess_face(face_crop_b)

    hog_a = _compute_hog_features(processed_a)
    hog_b = _compute_hog_features(processed_b)

    combined = _compute_similarity(processed_a, processed_b, hog_a, hog_b)
    match = combined >= FACE_MATCH_THRESHOLD

    return match, combined


def assign_person_ids(detection_results):
    known_signatures = {}
    person_images = {}
    next_person_id = 1

    for result in detection_results:
        img_path = result['path']
        for face in result['faces']:
            crop = face['crop']
            processed = _preprocess_face(crop)
            hog_vec = _compute_hog_features(processed)
            if hog_vec.size > 0:
                n = np.linalg.norm(hog_vec)
                if n > 0:
                    hog_vec = hog_vec / n

            matched_id = None
            best_similarity = 0

            for pid, sigs in known_signatures.items():
                for sig in sigs:
                    combined = _compute_similarity(
                        processed, sig['processed'],
                        hog_vec, sig['hog']
                    )
                    if combined > FACE_MATCH_THRESHOLD and combined > best_similarity:
                        best_similarity = combined
                        matched_id = pid

            if matched_id is None:
                matched_id = next_person_id
                next_person_id += 1
                known_signatures[matched_id] = []

            known_signatures[matched_id].append({
                'hog': hog_vec,
                'processed': processed
            })

            if matched_id not in person_images:
                person_images[matched_id] = set()
            person_images[matched_id].add(img_path)

    return {
        f"persona_{pid}": sorted(list(paths))
        for pid, paths in sorted(person_images.items())
    }
