import clip
import torch

def load_clip(model_name="ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, device