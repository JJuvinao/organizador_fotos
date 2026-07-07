"""
Extracción de embeddings faciales usando FaceNet.
Reemplaza el débil algoritmo HOG con deep learning.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import pickle
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE_DIR = Path(__file__).parent / 'embeddings_cache'
CACHE_DIR.mkdir(exist_ok=True)
EMBEDDING_DIM = 512

# Modelo global
_model = None

def get_model():
    """Obtiene o carga el modelo FaceNet una sola vez"""
    global _model
    if _model is None:
        try:
            from facenet_pytorch import InceptionResnetV1
            logger.info(f"Cargando FaceNet en {DEVICE}...")
            _model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
            logger.info("FaceNet cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando FaceNet: {e}")
            raise
    return _model


def _get_cache_path(image_path: str) -> Path:
    """Genera ruta de cache para un embedding"""
    cache_name = f"{hash(image_path)}.pkl"
    return CACHE_DIR / cache_name


def _normalize_face_image(face_crop: np.ndarray) -> torch.Tensor:
    """
    Normaliza y convierte una imagen de cara a tensor PyTorch.
    
    Args:
        face_crop: Imagen BGR de (H, W, 3) en rango [0, 255]
    
    Returns:
        Tensor normalizado de (1, 3, H, W)
    """
    # Convertir BGR a RGB
    rgb = face_crop[..., ::-1]
    
    # Convertir a PIL Image para resize consistente
    pil_img = PILImage.fromarray(rgb)
    pil_img = pil_img.resize((160, 160), PILImage.LANCZOS)
    
    # Convertir a tensor
    img_tensor = torch.tensor(np.array(pil_img), dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 160, 160)
    
    # Normalización ImageNet
    img_tensor = (img_tensor - 127.5) / 128.0
    
    return img_tensor.to(DEVICE)


def get_face_embedding(face_crop: np.ndarray, use_cache: bool = False, 
                       image_path: Optional[str] = None) -> np.ndarray:
    """
    Extrae embedding 512D de una cara usando FaceNet.
    
    Args:
        face_crop: Imagen BGR de cara de (H, W, 3)
        use_cache: Si True, intenta cargar del cache
        image_path: Ruta de imagen para cachear
    
    Returns:
        Embedding normalizado de forma (512,)
    """
    # Intenta cargar del cache
    if use_cache and image_path:
        cache_path = _get_cache_path(image_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error cargando cache {cache_path}: {e}")
    
    # Obtiene el modelo
    model = get_model()
    
    # Normaliza y procesa
    img_tensor = _normalize_face_image(face_crop)
    
    with torch.no_grad():
        embedding = model(img_tensor)
    
    embedding = embedding.cpu().numpy().squeeze()
    
    # Guarda en cache si es posible
    if use_cache and image_path:
        cache_path = _get_cache_path(image_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Error guardando cache {cache_path}: {e}")
    
    return embedding


def compute_face_distance(embedding_a: np.ndarray, 
                         embedding_b: np.ndarray) -> float:
    """
    Calcula distancia euclidiana entre dos embeddings.
    
    Args:
        embedding_a: Embedding de cara A
        embedding_b: Embedding de cara B
    
    Returns:
        Distancia euclidiana normalizada [0, 1]
    """
    distance = np.linalg.norm(embedding_a - embedding_b)
    # Normaliza a rango [0, 1] (max distancia es ~5 típicamente)
    return min(1.0, distance / 5.0)


def compute_batch_embeddings(face_crops: list, image_paths: Optional[list] = None,
                             use_cache: bool = True) -> list:
    """
    Computa embeddings para múltiples caras eficientemente.
    
    Args:
        face_crops: Lista de imágenes BGR de caras
        image_paths: Rutas correspondientes (para cache)
        use_cache: Si usar cache
    
    Returns:
        Lista de embeddings (512,)
    """
    embeddings = []
    model = get_model()
    
    for i, face_crop in enumerate(face_crops):
        image_path = image_paths[i] if image_paths else None
        
        # Intenta cache primero
        if use_cache and image_path:
            cache_path = _get_cache_path(image_path)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        embeddings.append(pickle.load(f))
                        continue
                except:
                    pass
        
        # Computa embedding
        embedding = get_face_embedding(face_crop, use_cache=False)
        embeddings.append(embedding)
    
    return embeddings


def clear_embeddings_cache():
    """Limpia todos los embeddings en cache"""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        logger.info("Cache de embeddings limpiado")
