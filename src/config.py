"""
Configuración centralizada de parámetros de detección y matching.
Permite ajustar sensibilidad del sistema.
"""

# ==================== DETECCIÓN DE ROSTROS ====================
# Umbral mínimo de confianza para considerar una detección válida (0-1)
# Valor más bajo = detecta más rostros (incluyendo parciales)
FACE_DETECTION_CONFIDENCE = 0.2  # Antes era 0.3

# Margen alrededor del rostro detectado (% del ancho/alto del rostro)
FACE_MARGIN_FACTOR = 0.25  # Antes era 0.2

# Umbral de IoU para considerar dos detecciones como duplicadas (0-1)
# Valor más alto = menos eliminación de duplicados
DUPLICATE_IOU_THRESHOLD = 0.4  # Antes era 0.5

# ==================== CALIDAD DE ROSTROS ====================
# Puntuación mínima de calidad para pasar al clustering (0-1)
# Valor más bajo = acepta más rostros de baja calidad
MIN_QUALITY_SCORE = 0.15  # Antes era 0.3

# Tamaño mínimo de rostro en pixels
MIN_FACE_SIZE = 32  # Antes era 64

# Umbrales de brillo
MIN_BRIGHTNESS = 10  # Más tolerante con rostros oscuros
MAX_BRIGHTNESS = 245  # Más tolerante con rostros brillantes

# ==================== MATCHING Y CLUSTERING ====================
# Distancia máxima para considerar dos rostros como la misma persona
# Valor más alto = agrupa más agresivamente
FACE_DISTANCE_THRESHOLD = 0.75  # Antes era 0.6

# Distancia máxima para fusionar clusters automáticamente
# Valor más alto = fusiona más clusters
CLUSTER_MERGE_THRESHOLD = 0.70  # Antes era 0.55

# Método de clustering jerárquico
# Opciones: 'ward', 'complete', 'average', 'single'
CLUSTERING_METHOD = 'average'  # Más tolerante que 'ward'

# ==================== EMBEDDINGS ====================
# Tamaño de las imágenes normalizadas (debe ser 160x160 para FaceNet)
EMBEDDING_IMAGE_SIZE = 160

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"
VERBOSE = True  # Más información de debugging
