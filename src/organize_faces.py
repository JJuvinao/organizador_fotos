import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_detector import process_folder
from src.face_matcher import assign_person_ids
from src.face_organizer import organize_faces, OUTPUT_BASE


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m src.organize_faces RUTA_CARPETA")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not Path(folder_path).exists():
        print(f"Error: La carpeta '{folder_path}' no existe.")
        sys.exit(1)

    print(f"Escaneando {folder_path} ...")
    detection_results, no_face_images, errors = process_folder(folder_path)

    if errors:
        print(f"\nAdvertencias ({len(errors)}):")
        for err in errors[:10]:
            print(f"  ! {err}")
        if len(errors) > 10:
            print(f"  ... y {len(errors) - 10} más")

    total = len(detection_results) + len(no_face_images)
    print(f"\nImágenes encontradas: {total}")
    print(f"  Con rostros: {len(detection_results)}")
    print(f"  Sin rostro:   {len(no_face_images)}")

    if total == 0:
        print("No se encontraron imágenes para procesar.")
        sys.exit(0)

    if detection_results:
        print("\nAgrupando rostros por persona ...")
        person_groups = assign_person_ids(detection_results)
        print(f"Personas distintas detectadas: {len(person_groups)}")
    else:
        person_groups = {}

    print(f"\nOrganizando en: {OUTPUT_BASE}")
    summary = organize_faces(person_groups, no_face_images)

    print("\nResumen:")
    for folder, info in summary.items():
        print(f"  {folder}/  →  {info['count']} foto(s)")

    print(f"\n¡Listo! Revisa: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
