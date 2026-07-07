import shutil
from pathlib import Path

OUTPUT_BASE = Path(r"C:\programas\fotos organizadas")


def _copy_to_folder(src_path, dst_dir):
    src = Path(src_path)
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        counter = 1
        while dst.exists():
            dst = dst_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    shutil.copy2(str(src), str(dst))
    return str(dst)


def get_next_person_number(output_dir):
    max_num = 0
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir() and item.name.startswith("persona_"):
                try:
                    num = int(item.name.split("_")[1])
                    max_num = max(max_num, num)
                except (IndexError, ValueError):
                    pass
    return max_num


def organize_faces(person_groups, no_face_images, output_base=None):
    if output_base is None:
        output_base = OUTPUT_BASE

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    next_num = get_next_person_number(output_base)
    summary = {}

    sorted_groups = sorted(
        person_groups.items(),
        key=lambda x: int(x[0].split("_")[1])
    )

    for _, image_paths in sorted_groups:
        next_num += 1
        folder_name = f"persona_{next_num}"
        person_dir = output_base / folder_name

        copied = []
        for src_path in image_paths:
            result = _copy_to_folder(src_path, person_dir)
            if result:
                copied.append(result)

        if copied:
            summary[folder_name] = {
                "count": len(copied),
                "files": copied
            }

    if no_face_images:
        no_face_dir = output_base / "sin_rostro"
        copied = []
        for src_path in no_face_images:
            result = _copy_to_folder(src_path, no_face_dir)
            if result:
                copied.append(result)
        if copied:
            summary["sin_rostro"] = {
                "count": len(copied),
                "files": copied
            }

    return summary
