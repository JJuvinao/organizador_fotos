"""
Pipeline mejorado de detección y organización de rostros.
Integra: detección → calidad → embeddings → clustering jerárquico → organización.
"""
import sys
import json
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_detector import process_folder
from src.face_quality import filter_low_quality_faces
from src.face_embeddings import compute_batch_embeddings
from src.face_clustering import assign_person_ids_hierarchical, get_clustering_stats
from src.face_organizer import organize_faces, OUTPUT_BASE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m src.organize_faces_improved RUTA_CARPETA [--quality-threshold 0.3]")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    quality_threshold = 0.3
    
    # Parsea argumentos opcionales
    if '--quality-threshold' in sys.argv:
        idx = sys.argv.index('--quality-threshold')
        if idx + 1 < len(sys.argv):
            quality_threshold = float(sys.argv[idx + 1])
    
    if not Path(folder_path).exists():
        print(f"Error: La carpeta '{folder_path}' no existe.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("ORGANIZADOR DE FOTOS - VERSIÓN MEJORADA CON DEEP LEARNING")
    print("="*60)
    
    # PASO 1: Detección inicial
    print(f"\n[1/5] Escaneando y detectando rostros en: {folder_path}")
    detection_results, no_face_images, errors = process_folder(folder_path)
    
    if errors:
        print(f"\n⚠️  Advertencias durante detección ({len(errors)}):")
        for err in errors[:5]:
            print(f"    • {err}")
        if len(errors) > 5:
            print(f"    ... y {len(errors) - 5} más")
    
    total_images = len(detection_results) + len(no_face_images)
    print(f"\n✓ Imágenes encontradas: {total_images}")
    print(f"  • Con rostros detectados: {len(detection_results)}")
    print(f"  • Sin rostros: {len(no_face_images)}")
    
    if total_images == 0:
        print("\n❌ No se encontraron imágenes para procesar.")
        sys.exit(0)
    
    # PASO 2: Filtrado de calidad
    if detection_results:
        print(f"\n[2/5] Filtrando rostros de baja calidad (threshold: {quality_threshold})...")
        detection_results, filtered_out = filter_low_quality_faces(
            detection_results, 
            quality_threshold=quality_threshold
        )
        
        if filtered_out:
            print(f"⚠️  {len(filtered_out)} rostros filtrados por baja calidad")
    
    if not detection_results:
        print("\n❌ No hay rostros de calidad suficiente para procesar.")
        sys.exit(0)
    
    # PASO 3: Extracción de embeddings
    print(f"\n[3/5] Extrayendo embeddings con FaceNet (puede tomar tiempo)...")
    embeddings_dict = {}
    
    for result in detection_results:
        img_path = result['path']
        face_crops = [face['crop'] for face in result['faces']]
        
        embeddings = compute_batch_embeddings(face_crops, [img_path] * len(face_crops), use_cache=True)
        
        # Usa el embedding del primer rostro (típicamente el principal)
        if embeddings:
            embeddings_dict[img_path] = embeddings[0]
    
    print(f"✓ {len(embeddings_dict)} embeddings extraídos")
    
    # PASO 4: Clustering jerárquico
    print(f"\n[4/5] Agrupando rostros usando clustering jerárquico...")
    person_groups = assign_person_ids_hierarchical(
        detection_results,
        embeddings_dict,
        distance_threshold=0.6,
        merge_threshold=0.55
    )
    
    # Estadísticas de clustering
    cluster_embeddings = {}
    for person_id, image_paths in person_groups.items():
        embeddings_list = [embeddings_dict[p] for p in image_paths if p in embeddings_dict]
        if embeddings_list:
            cluster_embeddings[person_id] = embeddings_list
    
    stats = get_clustering_stats(person_groups, cluster_embeddings)
    
    print(f"✓ Personas identificadas: {stats['total_personas']}")
    print(f"  • Promedio de fotos por persona: {stats['average_images_per_person']:.1f}")
    print(f"  • Distribución:")
    for size_key, count in sorted(stats['personas_by_size'].items()):
        print(f"    - {count} persona(s) con {size_key}")
    
    # PASO 5: Organización en carpetas
    print(f"\n[5/5] Organizando archivos en: {OUTPUT_BASE}")
    summary = organize_faces(person_groups, no_face_images)
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    
    for folder, info in summary.items():
        if folder == "sin_rostro":
            print(f"📁 {folder:20} → {info['count']:4} foto(s) sin rostro")
        else:
            confidence = stats.get('cluster_confidence', {}).get(folder, 'N/A')
            if isinstance(confidence, float):
                confidence_str = f"({confidence:.1%} confianza)"
            else:
                confidence_str = ""
            print(f"👤 {folder:20} → {info['count']:4} foto(s) {confidence_str}")
    
    total_organized = sum(v['count'] for v in summary.values())
    print(f"\n✅ Total organizado: {total_organized} foto(s)")
    print(f"📂 Ubicación: {OUTPUT_BASE}")
    print("\n¡Listo! Abre la carpeta para revisar los resultados.\n")


if __name__ == "__main__":
    main()
