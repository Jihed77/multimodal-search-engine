# 🔍 Multimodal AI Search Engine

## Text → Image | Image → Image Semantic Retrieval System

📸 Screenshots

Text Search Example
https://images/text_search.png

Image Search Example
https://images/image_search.png

You can also include a GIF of the full workflow.
See assets/demo.gif.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![AI](https://img.shields.io/badge/AI-Computer%20Vision-green.svg)
![CLIP](https://img.shields.io/badge/Model-OpenAI%20CLIP-purple.svg)
![FAISS](https://img.shields.io/badge/Search-FAISS-orange.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

A powerful **multimodal image search engine** that understands both **text descriptions** and **visual inputs**.

The system uses **OpenAI CLIP** to generate semantic embeddings and **FAISS** to perform fast vector similarity search.

Users can:

* 🔤 Search images using natural language (**Text → Image**)
* 🖼️ Upload an image and find similar images (**Image → Image**)

![Demo](demo.gif)

---

# 🚀 Project Overview

Traditional image search relies on keywords or manual labels.

This project introduces a semantic search approach:

```
User Query
     |
     ↓
CLIP Encoder
     |
     ↓
Vector Embedding
     |
     ↓
FAISS Similarity Search
     |
     ↓
Relevant Images
```

Instead of matching filenames, the system understands the **meaning** of images and text.

Example:

Input:

```
"a dog playing in snow"
```

Output:

Relevant images containing dogs and snow scenes.

---

# ✨ Features

## 🔤 Text-to-Image Search

Search images using natural language.

Examples:

```
a red car
a bird flying
a wooden chair
a person skiing
```

CLIP converts the text query into an embedding and retrieves visually relevant images.

---

## 🖼️ Image-to-Image Search

Upload an image and retrieve visually similar images.

The model captures:

* object similarity
* visual concepts
* semantic relationships

---

## ⚡ Fast Vector Search

Powered by:

**FAISS (Facebook AI Similarity Search)**

Advantages:

* optimized nearest-neighbor search
* scalable indexing
* fast retrieval
* efficient vector comparison

---

## 🧠 Multimodal Understanding

Using:

**OpenAI CLIP ViT-B/32**

CLIP creates a shared embedding space:

```
        Text

         ↓

      CLIP Space

         ↑

       Image
```

Images and text with similar meanings are placed close together.

---

# 📚 Dataset

## Caltech-256 Image Dataset

This project uses the **Caltech-256 Image Dataset**, a benchmark dataset for computer vision and image recognition tasks.

Dataset:

https://www.kaggle.com/datasets/jessicali9530/caltech256

Original source:

Caltech Vision Lab
http://www.vision.caltech.edu/Image_Datasets/Caltech256/

---

## Dataset Statistics

| Property                | Value              |
| ----------------------- | ------------------ |
| Total images            | 30,607             |
| Categories              | 257 object classes |
| Minimum images/category | 80                 |
| Median images/category  | 100                |
| Mean images/category    | 119                |
| Maximum images/category | 827                |

Examples of categories:

```
airplanes
backpacks
cars
frogs
cell phones
sailboats
chairs
tuning forks
```

---

## Why Caltech-256?

Compared to Caltech-101, it provides:

* more categories
* larger image collections
* higher visual diversity
* more challenging recognition tasks

In this project, the dataset is used for:

✅ image embedding generation
✅ semantic retrieval
✅ similarity search benchmarking

---

# 🏗️ System Architecture

```

                 User

                  |

        ----------------------

        |                    |

   Text Query          Image Upload

        |                    |

        ↓                    ↓


              OpenAI CLIP

                  |

                  ↓

          Image/Text Embeddings

                  |

                  ↓

              FAISS Index

                  |

                  ↓

          Similarity Ranking

                  |

                  ↓

          Retrieved Images


```

---

# 📁 Project Structure

```
multimodal-search-engine/

│                 
│
├── embeddings/
│   └── image_embeddings.npy     
│
├── index/
│   ├── faiss.index              
│   └── paths.pkl                
│
├── src/
│   ├── __init__.py
│   ├── clip_model.py            
│   ├── indexing.py              
│   ├── search.py                
│   └── utils.py                 
│
├── app.py                       
├── build_index.py               
├── requirements.txt             
└── README.md

```

---

# 🛠️ Tech Stack

| Component               | Technology  |
| ----------------------- | ----------- |
| Programming Language    | Python      |
| Multimodal Model        | OpenAI CLIP |
| Deep Learning Framework | PyTorch     |
| Vector Database         | FAISS       |
| Web Application         | Streamlit   |
| Image Processing        | Pillow      |
| Dataset                 | Caltech-256 |
| Data Processing         | NumPy       |

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/multimodal-search-engine.git

cd multimodal-search-engine
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For Windows:

```
faiss-cpu
```

For NVIDIA GPU:

```
faiss-gpu
```

---

# 📦 Build Search Index

Run:

```bash
python build_index.py
```

The script automatically:

✅ Downloads Caltech-256 dataset
✅ Loads CLIP model
✅ Encodes all images
✅ Generates embeddings
✅ Creates FAISS index
✅ Saves image mapping

Generated files:

```
embeddings/image_embeddings.npy

index/faiss.index

index/paths.pkl
```

---

# ▶️ Run Application

Launch Streamlit:

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

# 🖥️ How To Use

## Text Search

1. Select:

```
Text Search
```

2. Enter:

```
a red sports car
```

3. Get:

* top matching images
* similarity scores

---

## Image Search

1. Select:

```
Image Search
```

2. Upload:

```
.jpg
.jpeg
.png
```

3. Retrieve similar images.

---



Built with ❤️ for an AI & Data Science portfolio 🚀
