import torch
import clip
from PIL import Image
import numpy as np

def search_text(query, model, preprocess, device, index, valid_paths, top_k=5):
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    text_features = text_features.cpu().numpy().astype('float32')
    scores, indices = index.search(text_features, top_k)
    results = [(valid_paths[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
    return results

def search_image(query_image_path, model, preprocess, device, index, valid_paths, top_k=5):
    img = Image.open(query_image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
    img_features = img_features / img_features.norm(dim=1, keepdim=True)
    img_features = img_features.cpu().numpy().astype('float32')
    scores, indices = index.search(img_features, top_k)
    results = [(valid_paths[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
    return results