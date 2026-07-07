"""
API mejorada con endpoints para correcciones manuales de agrupamientos.
Permite fusionar y dividir personas después del clustering automático.
"""
import sys
import os
import json
from pathlib import Path
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify, render_template
import shutil

from src.face_detector import process_folder, scan_images, IMAGE_EXTENSIONS
from src.face_quality import filter_low_quality_faces
from src.face_embeddings import compute_batch_embeddings, clear_embeddings_cache
from src.face_clustering import assign_person_ids_hierarchical, get_clustering_stats
from src.face_organizer import organize_faces, OUTPUT_BASE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Estado global para la sesión actual
current_session = {
    'detection_results': None,
    'no_face_images': None,
    'embeddings_dict': None,
    'person_groups': None,
    'cluster_stats': None,
    'folder_path': None,
}


@app.route("/")
def index():
    return render_template("index_improved.html", output_path=str(OUTPUT_BASE))


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Escanea una carpeta y cuenta imágenes"""
    data = request.get_json()
    folder_path = data.get("folder_path", "").strip()

    if not folder_path:
        return jsonify({"error": "Debes proporcionar la ruta de una carpeta."}), 400

    if not Path(folder_path).exists():
        return jsonify({"error": f"La carpeta '{folder_path}' no existe."}), 400

    try:
        scanned = scan_images(folder_path)
        return jsonify({
            "total": len(scanned),
            "ruta": folder_path,
            "extensiones": sorted(list(IMAGE_EXTENSIONS))
        })
    except Exception as e:
        return jsonify({"error": f"Error al escanear: {str(e)}"}), 500


@app.route("/api/organize", methods=["POST"])
def api_organize():
    """
    Organiza fotos: detección → calidad → embeddings → clustering.
    Guarda estado para correcciones posteriores.
    """
    data = request.get_json()
    folder_path = data.get("folder_path", "").strip()
    quality_threshold = float(data.get("quality_threshold", 0.3))

    if not folder_path:
        return jsonify({"error": "Debes proporcionar la ruta de una carpeta."}), 400

    if not Path(folder_path).exists():
        return jsonify({"error": f"La carpeta '{folder_path}' no existe."}), 400

    try:
        logger.info(f"[1/5] Escaneando {folder_path}...")
        scanned = scan_images(folder_path)
        
        if len(scanned) == 0:
            return jsonify({
                "error": "No se encontraron archivos de imagen en la carpeta.",
                "ruta_buscada": folder_path,
                "extensiones_soportadas": sorted(list(IMAGE_EXTENSIONS))
            }), 400

        logger.info(f"[2/5] Detectando rostros...")
        detection_results, no_face_images, errors = process_folder(folder_path)

        if not detection_results and not no_face_images:
            msg = "No se pudieron procesar las imágenes."
            if errors:
                msg += f" ({len(errors)} errores)"
            return jsonify({
                "error": msg,
                "errores_detalle": errors[:20] if errors else None,
                "total_escaneadas": len(scanned)
            }), 400

        logger.info(f"[3/5] Filtrando calidad...")
        detection_results, filtered_out = filter_low_quality_faces(
            detection_results,
            quality_threshold=quality_threshold
        )

        if not detection_results:
            return jsonify({
                "error": "No hay rostros de calidad suficiente después del filtrado.",
                "filtered_out": len(filtered_out)
            }), 400

        logger.info(f"[4/5] Extrayendo embeddings...")
        embeddings_dict = {}
        for result in detection_results:
            img_path = result['path']
            face_crops = [face['crop'] for face in result['faces']]
            embeddings = compute_batch_embeddings(face_crops, [img_path] * len(face_crops), use_cache=True)
            
            if embeddings:
                embeddings_dict[img_path] = embeddings[0]

        logger.info(f"[5/5] Clustering jerárquico...")
        person_groups = assign_person_ids_hierarchical(
            detection_results,
            embeddings_dict,
            distance_threshold=0.6,
            merge_threshold=0.55
        )

        # Calcula estadísticas
        cluster_embeddings = {}
        for person_id, image_paths in person_groups.items():
            embeddings_list = [embeddings_dict[p] for p in image_paths if p in embeddings_dict]
            if embeddings_list:
                cluster_embeddings[person_id] = embeddings_list

        stats = get_clustering_stats(person_groups, cluster_embeddings)

        # Guarda estado para correcciones posteriores
        current_session['detection_results'] = detection_results
        current_session['no_face_images'] = no_face_images
        current_session['embeddings_dict'] = embeddings_dict
        current_session['person_groups'] = person_groups
        current_session['cluster_stats'] = stats
        current_session['folder_path'] = folder_path

        total_con_rostro = sum(len(paths) for paths in person_groups.values())
        total_sin_rostro = len(no_face_images)

        return jsonify({
            "total_fotos": len(no_face_images) + len(detection_results),
            "fotos_con_rostro": total_con_rostro,
            "sin_rostro": total_sin_rostro,
            "personas": len(person_groups),
            "personas_list": list(person_groups.keys()),
            "output_path": str(OUTPUT_BASE),
            "stats": stats,
            "advertencias": errors if errors else None,
            "filtered_quality": len(filtered_out)
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


@app.route("/api/person_details/<person_id>", methods=["GET"])
def get_person_details(person_id):
    """Obtiene detalles de una persona y sus imágenes"""
    if not current_session['person_groups']:
        return jsonify({"error": "No hay sesión activa"}), 400

    person_groups = current_session['person_groups']
    embeddings_dict = current_session['embeddings_dict']

    if person_id not in person_groups:
        return jsonify({"error": f"Persona {person_id} no encontrada"}), 404

    image_paths = person_groups[person_id]
    
    # Obtiene embeddings
    embeddings = [embeddings_dict.get(p) for p in image_paths if p in embeddings_dict]
    
    return jsonify({
        "person_id": person_id,
        "image_count": len(image_paths),
        "images": image_paths,
        "embedding_quality": current_session['cluster_stats'].get('cluster_confidence', {}).get(person_id, 'N/A')
    })


@app.route("/api/merge_persons", methods=["POST"])
def merge_persons():
    """Fusiona dos personas manualmente"""
    if not current_session['person_groups']:
        return jsonify({"error": "No hay sesión activa"}), 400

    data = request.get_json()
    person_a = data.get("person_a")
    person_b = data.get("person_b")

    if not person_a or not person_b:
        return jsonify({"error": "Debes proporcionar dos personas"}), 400

    person_groups = current_session['person_groups']

    if person_a not in person_groups or person_b not in person_groups:
        return jsonify({"error": "Una o ambas personas no existen"}), 404

    try:
        # Fusiona
        person_groups[person_a].extend(person_groups[person_b])
        person_groups[person_a] = sorted(list(set(person_groups[person_a])))
        del person_groups[person_b]

        logger.info(f"Fusionadas {person_a} y {person_b}")

        return jsonify({
            "success": True,
            "message": f"Fusionadas {person_a} y {person_b}",
            "new_count": len(person_groups[person_a])
        })
    except Exception as e:
        return jsonify({"error": f"Error al fusionar: {str(e)}"}), 500


@app.route("/api/split_person", methods=["POST"])
def split_person():
    """Divide una persona en dos basado en índices de imágenes"""
    if not current_session['person_groups']:
        return jsonify({"error": "No hay sesión activa"}), 400

    data = request.get_json()
    person_id = data.get("person_id")
    split_indices = data.get("split_indices", [])  # Índices a mover a nueva persona

    if not person_id or not split_indices:
        return jsonify({"error": "Parámetros inválidos"}), 400

    person_groups = current_session['person_groups']

    if person_id not in person_groups:
        return jsonify({"error": "Persona no encontrada"}), 404

    try:
        images = person_groups[person_id]
        
        # Valida índices
        split_indices = [int(i) for i in split_indices if 0 <= int(i) < len(images)]
        
        if not split_indices:
            return jsonify({"error": "Índices inválidos"}), 400

        # Divide
        images_to_move = [images[i] for i in sorted(split_indices, reverse=True)]
        for i in sorted(split_indices, reverse=True):
            del person_groups[person_id][i]

        # Crea nueva persona
        max_id = max([int(p.split('_')[1]) for p in person_groups.keys() if p.startswith('persona_')])
        new_person_id = f"persona_{max_id + 1}"
        person_groups[new_person_id] = images_to_move

        logger.info(f"Dividida {person_id}: {len(images_to_move)} imágenes movidas a {new_person_id}")

        return jsonify({
            "success": True,
            "message": f"Persona dividida",
            "original_person": person_id,
            "original_count": len(person_groups[person_id]),
            "new_person": new_person_id,
            "new_count": len(person_groups[new_person_id])
        })
    except Exception as e:
        return jsonify({"error": f"Error al dividir: {str(e)}"}), 500


@app.route("/api/finalize", methods=["POST"])
def finalize_organization():
    """Finaliza la organización con los agrupamientos editados"""
    if not current_session['person_groups']:
        return jsonify({"error": "No hay sesión activa"}), 400

    try:
        person_groups = current_session['person_groups']
        no_face_images = current_session['no_face_images']

        logger.info("Finalizando organización...")
        summary = organize_faces(person_groups, no_face_images)

        # Limpia sesión
        current_session['detection_results'] = None
        current_session['embeddings_dict'] = None
        current_session['person_groups'] = None
        current_session['cluster_stats'] = None

        return jsonify({
            "success": True,
            "message": "Fotos organizadas exitosamente",
            "output_path": str(OUTPUT_BASE),
            "summary": summary
        })

    except Exception as e:
        logger.error(f"Error finalizando: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error al finalizar: {str(e)}"}), 500


@app.route("/api/clear_cache", methods=["POST"])
def api_clear_cache():
    """Limpia el cache de embeddings"""
    try:
        clear_embeddings_cache()
        return jsonify({"success": True, "message": "Cache limpiado"})
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    print(f"🚀 Servidor corriendo en http://localhost:5000")
    print(f"📂 Salida: {OUTPUT_BASE}")
    app.run(debug=True, port=5000)
