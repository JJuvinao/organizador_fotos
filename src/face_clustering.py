"""
Clustering jerárquico de rostros usando embeddings FaceNet.
Reemplaza el matching simple con un algoritmo robusto.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Parámetros de clustering
DEFAULT_DISTANCE_THRESHOLD = 0.6
MERGE_THRESHOLD = 0.55  # Para fusión de clusters similares


def cluster_embeddings_hierarchical(embeddings: np.ndarray,
                                   distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
                                   method: str = 'ward') -> np.ndarray:
    """
    Agrupa embeddings usando clustering jerárquico.
    
    Args:
        embeddings: Array de forma (n_faces, 512)
        distance_threshold: Umbral de distancia para formar clusters
        method: Método de linkage ('ward', 'complete', 'average', 'single')
    
    Returns:
        Array de cluster IDs de forma (n_faces,)
    """
    if len(embeddings) < 2:
        return np.array([1])
    
    # Calcula distancias
    distances = pdist(embeddings, metric='euclidean')
    
    # Clustering jerárquico
    linkage_matrix = linkage(distances, method=method)
    
    # Forma clusters
    clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    logger.info(f"Clustering jerárquico: {len(embeddings)} caras → {len(np.unique(clusters))} clusters")
    
    return clusters


def assign_person_ids_hierarchical(detection_results: list,
                                   embeddings_dict: Dict[str, np.ndarray],
                                   distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
                                   merge_threshold: float = MERGE_THRESHOLD) -> Dict[str, List[str]]:
    """
    Agrupa rostros detectados en personas usando clustering jerárquico.
    
    Args:
        detection_results: Resultados de face_detector.process_folder()
        embeddings_dict: Dict {image_path: embedding}
        distance_threshold: Umbral para clustering inicial
        merge_threshold: Umbral para fusionar clusters similares
    
    Returns:
        Dict {persona_N: [image_path1, image_path2, ...]}
    """
    if not detection_results:
        logger.warning("Sin resultados de detección para agrupar")
        return {}
    
    # Recolecta embeddings y mapeos
    face_data = []
    embeddings_list = []
    
    for result in detection_results:
        img_path = result['path']
        for face_idx, face in enumerate(result['faces']):
            embedding = embeddings_dict.get(img_path)
            if embedding is None:
                logger.warning(f"Embedding no encontrado para {img_path}")
                continue
            
            face_data.append({
                'image_path': img_path,
                'face_index': face_idx,
                'embedding': embedding
            })
            embeddings_list.append(embedding)
    
    if not embeddings_list:
        logger.warning("Sin embeddings para agrupar")
        return {}
    
    embeddings_array = np.array(embeddings_list)
    
    # Clustering jerárquico
    cluster_ids = cluster_embeddings_hierarchical(embeddings_array, distance_threshold)
    
    # Mapea a personas
    person_groups = {}
    cluster_embeddings = {}
    
    for face_idx, (data, cluster_id) in enumerate(zip(face_data, cluster_ids)):
        person_key = f"persona_{cluster_id}"
        
        if person_key not in person_groups:
            person_groups[person_key] = []
            cluster_embeddings[person_key] = []
        
        person_groups[person_key].append(data['image_path'])
        cluster_embeddings[person_key].append(data['embedding'])
    
    # Elimina duplicados de rutas (si una imagen tiene múltiples caras del mismo cluster)
    for person_id in person_groups:
        person_groups[person_id] = sorted(list(set(person_groups[person_id])))
    
    # Intenta fusionar clusters muy similares
    person_groups = merge_similar_clusters(person_groups, cluster_embeddings, merge_threshold)
    
    logger.info(f"Agrupamiento final: {len(person_groups)} personas identificadas")
    
    return person_groups


def merge_similar_clusters(person_groups: Dict[str, List[str]],
                          cluster_embeddings: Dict[str, List[np.ndarray]],
                          merge_threshold: float = MERGE_THRESHOLD) -> Dict[str, List[str]]:
    """
    Detecta y fusiona clusters que probablemente sean la misma persona.
    
    Args:
        person_groups: Dict de personas y sus imágenes
        cluster_embeddings: Dict de personas y sus embeddings
        merge_threshold: Umbral de distancia para fusionar
    
    Returns:
        Dict actualizado con clusters fusionados
    """
    if len(person_groups) < 2:
        return person_groups
    
    # Calcula centros de clusters
    cluster_centers = {}
    for person_id, embeddings_list in cluster_embeddings.items():
        if embeddings_list:
            center = np.mean(embeddings_list, axis=0)
            cluster_centers[person_id] = center
    
    # Encuentra pares que deben fusionarse
    to_merge = []
    person_ids = list(cluster_centers.keys())
    
    for i in range(len(person_ids)):
        for j in range(i + 1, len(person_ids)):
            center_i = cluster_centers[person_ids[i]]
            center_j = cluster_centers[person_ids[j]]
            distance = np.linalg.norm(center_i - center_j)
            
            if distance < merge_threshold:
                logger.info(f"Fusionando {person_ids[i]} con {person_ids[j]} "
                           f"(distancia: {distance:.3f})")
                to_merge.append((person_ids[i], person_ids[j]))
    
    # Aplica fusiones
    for primary, secondary in to_merge:
        if primary in person_groups and secondary in person_groups:
            person_groups[primary].extend(person_groups[secondary])
            person_groups[primary] = sorted(list(set(person_groups[primary])))
            del person_groups[secondary]
    
    return person_groups


def compute_cluster_confidence(embeddings: np.ndarray) -> float:
    """
    Calcula confianza en un cluster basado en cohesión intra-cluster.
    Valores más altos = cluster más coherente.
    
    Args:
        embeddings: Array de embeddings en el cluster
    
    Returns:
        Confianza [0, 1]
    """
    if len(embeddings) < 2:
        return 1.0
    
    # Distancia promedio dentro del cluster
    distances = pdist(embeddings, metric='euclidean')
    mean_distance = np.mean(distances)
    
    # Normaliza (típicamente < 1.0 para mismo cluster)
    confidence = 1.0 - min(1.0, mean_distance / 2.0)
    
    return float(confidence)


def get_clustering_stats(person_groups: Dict[str, List[str]],
                        cluster_embeddings: Dict[str, List[np.ndarray]]) -> dict:
    """
    Calcula estadísticas del clustering.
    
    Args:
        person_groups: Dict de personas
        cluster_embeddings: Dict de embeddings por persona
    
    Returns:
        Dict con estadísticas
    """
    stats = {
        'total_personas': len(person_groups),
        'personas_by_size': {},
        'cluster_confidence': {},
        'average_images_per_person': 0,
    }
    
    total_images = 0
    for person_id, images in person_groups.items():
        n_images = len(images)
        total_images += n_images
        
        size_key = f"{n_images}_images"
        stats['personas_by_size'][size_key] = stats['personas_by_size'].get(size_key, 0) + 1
        
        # Confianza del cluster
        if person_id in cluster_embeddings and cluster_embeddings[person_id]:
            embeddings = np.array(cluster_embeddings[person_id])
            confidence = compute_cluster_confidence(embeddings)
            stats['cluster_confidence'][person_id] = float(confidence)
    
    if total_images > 0:
        stats['average_images_per_person'] = total_images / len(person_groups)
    
    return stats
