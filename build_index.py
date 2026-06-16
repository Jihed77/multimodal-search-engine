from src.clip_model import load_clip
from src.indexing import build_index
from src.utils import get_image_paths
import kagglehub
from src.utils import get_image_paths, deduplicate_paths_by_content


# Téléchargement Caltech256
print("Téléchargement du dataset...")
path = kagglehub.dataset_download("jessicali9530/caltech256")
data_dir = path + "/256_ObjectCategories"
print(f"Dataset dans : {data_dir}")

paths = get_image_paths(data_dir)
print(f"{len(paths)} images avant dédoublonnage")
paths = deduplicate_paths_by_content(paths)
print(f"{len(paths)} images après dédoublonnage")

# Chargement modèle CLIP
model, preprocess, device = load_clip()

# Construction index (embeddings + FAISS)
build_index(paths, model, preprocess, device)
print("Terminé !")