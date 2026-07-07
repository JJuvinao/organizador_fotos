import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify, render_template
from pathlib import Path

from src.face_detector import process_folder, scan_images, IMAGE_EXTENSIONS
from src.face_matcher import assign_person_ids
from src.face_organizer import organize_faces, OUTPUT_BASE

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", output_path=str(OUTPUT_BASE))


@app.route("/api/scan", methods=["POST"])
def api_scan():
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
    data = request.get_json()
    folder_path = data.get("folder_path", "").strip()

    if not folder_path:
        return jsonify({"error": "Debes proporcionar la ruta de una carpeta."}), 400

    if not Path(folder_path).exists():
        return jsonify({"error": f"La carpeta '{folder_path}' no existe."}), 400

    try:
        scanned = scan_images(folder_path)
        print(f"[DEBUG] Carpeta: {folder_path}")
        print(f"[DEBUG] Imágenes escaneadas: {len(scanned)}")

        if len(scanned) == 0:
            return jsonify({
                "error": "No se encontraron archivos de imagen en la carpeta.",
                "ruta_buscada": folder_path,
                "extensiones_soportadas": sorted(list(IMAGE_EXTENSIONS))
            }), 400

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

        if detection_results:
            person_groups = assign_person_ids(detection_results)
        else:
            person_groups = {}

        summary = organize_faces(person_groups, no_face_images)

        total_con_rostro = sum(
            v["count"] for k, v in summary.items() if k != "sin_rostro"
        )
        total_sin_rostro = summary.get("sin_rostro", {}).get("count", 0)

        return jsonify({
            "total_fotos": len(no_face_images) + len(detection_results),
            "fotos_con_rostro": total_con_rostro,
            "sin_rostro": total_sin_rostro,
            "personas": len(person_groups),
            "carpetas": list(summary.keys()),
            "output_path": str(OUTPUT_BASE),
            "detalle": summary,
            "advertencias": errors if errors else None
        })

    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


if __name__ == "__main__":
    print(f"Servidor corriendo en http://localhost:5000")
    app.run(debug=True, port=5000)
