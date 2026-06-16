import streamlit as st
from PIL import Image
import sys
sys.path.append('src')

from clip_model import load_clip
from indexing import load_index
from search import search_text, search_image

@st.cache_resource
def load_engine():
    model, preprocess, device = load_clip()
    index, valid_paths = load_index()
    return model, preprocess, device, index, valid_paths

model, preprocess, device, index, valid_paths = load_engine()

st.title("Moteur de recherche multimodal")
mode = st.radio("Mode de recherche", ("Texte", "Image"))

if mode == "Texte":
    query = st.text_input("Entrez votre description")
    if query:
        results = search_text(query, model, preprocess, device, index, valid_paths, top_k=5)
        cols = st.columns(3)
        for i, (path, score) in enumerate(results):
            img = Image.open(path)
            cols[i%3].image(img, caption=f"Score: {score:.2f}", use_column_width=True)

else:
    uploaded = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    if uploaded:
        # Sauvegarde temporaire
        with open("temp_query.jpg", "wb") as f:
            f.write(uploaded.getbuffer())
        st.image(uploaded, caption="Image requête", width=300)
        results = search_image("temp_query.jpg", model, preprocess, device, index, valid_paths, top_k=5)
        cols = st.columns(3)
        for i, (path, score) in enumerate(results):
            img = Image.open(path)
            cols[i%3].image(img, caption=f"Score: {score:.2f}", use_column_width=True)