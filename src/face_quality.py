"""
Evaluación de calidad de rostros detectados.
Filtra caras borrosas, pequeñas o de baja calidad.
"""
import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Thresholds de calidad
MIN_FACE_SIZE = 64  # pixels, mínimo para embedding útil
MIN_SHARPNESS = 100.0  # Laplacian variance
MIN_CONTRAST = 0.1  # Desviación estándar normalizada
MIN_BRIGHTNESS = 20  # No demasiado oscuro
MAX_BRIGHTNESS = 235  # No demasiado claro


def compute_sharpness(gray: np.ndarray) -> float:
    """
    Calcula nitidez usando varianza de Laplacian.
    Valores más altos = más nítido.
    
    Args:
        gray: Imagen en escala de grises
    
    Returns:
        Puntuación de nitidez (típicamente 50-500)
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return float(sharpness)


def compute_brightness(gray: np.ndarray) -> float:
    """
    Calcula brillo promedio de la imagen.
    
    Args:
        gray: Imagen en escala de grises
    
    Returns:
        Brillo promedio [0, 255]
    """
    return float(np.mean(gray))


def compute_contrast(gray: np.ndarray) -> float:
    """
    Calcula contraste como desviación estándar de píxeles.
    
    Args:
        gray: Imagen en escala de grises
    
    Returns:
        Contraste normalizado [0, 1]
    """
    contrast = float(np.std(gray)) / 255.0
    return contrast


def compute_face_blur(face_crop: np.ndarray) -> float:
    """
    Detecta si una cara está borrosa.
    Valores más bajos = más borrosa.
    
    Args:
        face_crop: Imagen BGR de cara
    
    Returns:
        Score de nitidez [0, 1]
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    sharpness = compute_sharpness(gray)
    
    # Normaliza a rango [0, 1]
    blur_score = min(1.0, sharpness / 200.0)
    return blur_score


def compute_face_quality(face_crop: np.ndarray, 
                        min_size: int = MIN_FACE_SIZE) -> Tuple[float, dict]:
    """
    Calcula puntuación de calidad integral de una cara.
    
    Args:
        face_crop: Imagen BGR de cara
        min_size: Tamaño mínimo requerido en pixels
    
    Returns:
        (quality_score [0, 1], metrics_dict)
    """
    h, w = face_crop.shape[:2]
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    
    metrics = {}
    
    # 1. Tamaño (40% del score)
    size_score = 1.0 if min(h, w) >= min_size else (min(h, w) / min_size) ** 2
    metrics['size_score'] = float(size_score)
    metrics['face_size'] = min(h, w)
    
    # 2. Nitidez (40% del score)
    sharpness = compute_sharpness(gray)
    sharpness_score = min(1.0, sharpness / MIN_SHARPNESS)
    metrics['sharpness_score'] = float(sharpness_score)
    metrics['sharpness'] = float(sharpness)
    
    # 3. Contraste (15% del score)
    contrast = compute_contrast(gray)
    contrast_score = min(1.0, contrast / 0.3)
    metrics['contrast_score'] = float(contrast_score)
    metrics['contrast'] = float(contrast)
    
    # 4. Brillo (5% del score - solo verificación)
    brightness = compute_brightness(gray)
    is_too_dark = brightness < MIN_BRIGHTNESS
    is_too_bright = brightness > MAX_BRIGHTNESS
    brightness_ok = 1.0 if not (is_too_dark or is_too_bright) else 0.5
    metrics['brightness_score'] = float(brightness_ok)
    metrics['brightness'] = float(brightness)
    
    # Score combinado con pesos
    quality_score = (
        size_score * 0.40 +
        sharpness_score * 0.40 +
        contrast_score * 0.15 +
        brightness_ok * 0.05
    )
    
    metrics['overall_quality'] = float(quality_score)
    
    return float(quality_score), metrics


def filter_low_quality_faces(detection_results: list, 
                             quality_threshold: float = 0.3) -> Tuple[list, list]:
    """
    Filtra rostros de baja calidad de los resultados de detección.
    
    Args:
        detection_results: Resultados de face_detector.process_folder()
        quality_threshold: Score mínimo de calidad [0, 1]
    
    Returns:
        (high_quality_results, filtered_out_faces)
    """
    high_quality = []
    filtered_out = []
    
    for result in detection_results:
        img_path = result['path']
        original_faces = result['faces']
        quality_faces = []
        
        for face in original_faces:
            crop = face['crop']
            quality_score, metrics = compute_face_quality(crop)
            
            if quality_score >= quality_threshold:
                face['quality_score'] = quality_score
                face['quality_metrics'] = metrics
                quality_faces.append(face)
            else:
                filtered_out.append({
                    'image_path': img_path,
                    'quality_score': quality_score,
                    'metrics': metrics,
                    'reason': 'Low quality face'
                })
        
        if quality_faces:
            high_quality.append({
                'path': img_path,
                'faces': quality_faces
            })
    
    logger.info(f"Filtradas {len(filtered_out)} caras de baja calidad "
                f"({len(high_quality)} imágenes con buenas caras)")
    
    return high_quality, filtered_out
