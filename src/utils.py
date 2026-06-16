import hashlib
from pathlib import Path

def get_image_paths(data_dir):
    data_path = Path(data_dir)
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for ext in extensions:
        paths.extend(data_path.rglob(ext))
    return list(set(paths))

def deduplicate_paths_by_content(paths):
    """Élimine les fichiers qui ont le même contenu (hash md5)."""
    seen = set()
    unique = []
    for p in paths:
        try:
            with open(p, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except:
            continue
        if file_hash not in seen:
            seen.add(file_hash)
            unique.append(p)
    return unique