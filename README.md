# 🖼️ Image Caption Generator using Deep Learning

## 📌 Overview
This project implements an **Image Caption Generator** that automatically generates meaningful natural language descriptions for images.  
It combines **Computer Vision** and **Natural Language Processing (NLP)** using a deep learning–based **encoder–decoder architecture**.

The model uses a pretrained **CNN (Xception)** to extract visual features from images and an **LSTM-based decoder** to generate captions word by word.

---

## 🚀 Features
- Generates human-like captions for input images
- Uses **pretrained Xception CNN** for robust image feature extraction
- Sequence-to-sequence caption generation using **LSTM**
- Custom **data generator** for memory-efficient training
- Supports **Greedy decoding** (Beam Search can be extended)
- Compatible with COCO dataset format
- Modular and extensible design

---

## 🧠 Model Architecture

### Encoder (Image Feature Extractor)
- Pretrained **Xception** model (ImageNet weights)
- Final classification layer removed
- Global Average Pooling → 2048-dimensional feature vector
- Dense layer to reduce dimensionality

### Decoder (Caption Generator)
- Word embeddings for captions
- LSTM to capture linguistic context
- Image features and text features merged using element-wise addition
- Softmax layer predicts the next word in the sequence

**Inputs:**
- Image feature vector  
- Partial caption sequence
- <img width="970" height="647" alt="image" src="https://github.com/user-attachments/assets/e2c679fc-fffa-4573-a748-77c288f26c34" />
- A boy is riding a bicycle on the road.

**Output:**
- Next predicted word

---

## 📂 Dataset
- **COCO (Common Objects in Context) Dataset**
- Each image has **multiple human-written captions**
- COCO API (`pycocotools`) used for loading images and annotations

Files used:
- `captions_train2017.json`
- `instances_train2017.json`
- COCO train & validation images

---

## 🔄 Data Preprocessing

### Image Preprocessing
- Resize images to **299 × 299**
- Normalize pixel values to range `[-1, 1]`
- Extract 2048-dim features using Xception

### Caption Preprocessing
- Convert text to lowercase
- Remove punctuation and special characters
- Replace hyphens with spaces
- Add `<start>` and `<end>` tokens
- Tokenize captions using Keras `Tokenizer`
- Pad sequences to maximum caption length

---

## ⚙️ Training Strategy
- Loss function: **Categorical Cross-Entropy**
- Optimizer: **Adam**
- Custom Python generator used to:
  - Reduce memory usage
  - Dynamically generate `(image, caption) → next word` pairs
- Dropout layers added to reduce overfitting

---

## 🧪 Inference Pipeline
1. Input image passed through Xception to extract features
2. Caption generation starts with `<start>` token
3. Model predicts next word iteratively
4. Stops when `<end>` token or max length is reached

---

## 📊 Evaluation
- Qualitative evaluation by visual inspection of generated captions
- BLEU score evaluation can be added for quantitative analysis
- Greedy decoding implemented (Beam Search supported)

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- Xception CNN
- LSTM
- NumPy
- NLTK
- Matplotlib
- COCO API (pycocotools)

---

## 📁 Project Structure

```text
image-caption-generator/
│
├── notebooks/
│   ├── image_caption_generator.ipynb
│   └── exploratory_analysis.ipynb
│
├── data/                     # Dataset directory (not tracked in Git)
│   ├── images/
│   └── captions.txt
│
├── models/                   # Saved models & checkpoints (ignored)
│   └── caption_model.h5
│
├── src/                      # Source code (optional refactor)
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   └── inference.py
│
├── outputs/
│   └── sample_predictions.png
│
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```
---

## 🌍 Real-World Applications
- Assistive technology for visually impaired users
- Automatic alt-text generation for accessibility
- Image search and content indexing
- Social media photo tagging
- E-commerce product description generation
- Surveillance and security reporting

---

## 🧠 Challenges Faced & Learnings
- **Poor initial caption quality** → improved via better preprocessing and CNN selection
- **Large dataset memory constraints** → solved using data generators
- **Sequence alignment issues** → resolved by careful input-output pairing
- Strong hands-on experience in **multi-modal deep learning**

---

## 🔮 Future Improvements
- Integrate **attention mechanism**
- Implement **beam search decoding**
- Explore **Transformer-based architectures**
- Add BLEU, METEOR, CIDEr evaluation
- Fine-tune CNN layers for domain-specific data

---

## 👩‍💻 Author
**Swati Mittal**  
B.Tech Computer Science  
Deep Learning | Computer Vision | NLP  

---

## ⭐ Acknowledgements
- COCO Dataset
- TensorFlow & Keras community
- Research work on image captioning and encoder–decoder models

