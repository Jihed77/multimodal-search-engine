import torch
import faiss
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
import os

def encode_images_batch(paths, model, preprocess, device, batch_size=64):
    embeddings = []
    valid_paths = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_imgs = []
        batch_valid = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(preprocess(img))
                batch_valid.append(p)
            except Exception as e:
                print(f"Skipping {p}: {e}")
        if not batch_imgs:
            continue
        imgs = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            emb = model.encode_image(imgs)
        emb = emb / emb.norm(dim=1, keepdim=True)
        embeddings.append(emb.cpu())
        valid_paths.extend(batch_valid)
        print(f"Processed {len(valid_paths)}/{len(paths)} images", end="\r")
    all_emb = torch.cat(embeddings, dim=0).numpy().astype('float32')
    return all_emb, valid_paths

def build_index(paths, model, preprocess, device, save_dir="index", emb_dir="embeddings"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    print("Encoding images...")
    embeddings, valid_paths = encode_images_batch(paths, model, preprocess, device)

    # Sauvegarde embeddings
    np.save(os.path.join(emb_dir, "image_embeddings.npy"), embeddings)
    print("Embeddings saved.")

    # Construction index FAISS
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # Sauvegarde index et paths
    faiss.write_index(index, os.path.join(save_dir, "faiss.index"))
    with open(os.path.join(save_dir, "paths.pkl"), "wb") as f:
        pickle.dump(valid_paths, f)
    print(f"Index saved with {index.ntotal} vectors.")
    return index, valid_paths

def load_index(index_dir="index", emb_dir="embeddings"):
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "paths.pkl"), "rb") as f:
        valid_paths = pickle.load(f)
    return index, valid_paths