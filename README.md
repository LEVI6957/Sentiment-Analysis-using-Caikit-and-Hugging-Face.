<h1 align="center">🌟 Sentiment Analysis using Caikit and Hugging Face 🌟</h1>
<p align="center">🔍 Explore Sentiment Analysis with Caikit and Hugging Face! 🔍</p>

<div align="center">
    <img src="https://img.shields.io/badge/Jupyter-FFAA00?style=for-the-badge&logo=Jupyter&logoColor=white">
    <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=Python&logoColor=ffdd54">
    <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white">
    <img src="https://img.shields.io/badge/Infinite_Learning-4B0082?style=for-the-badge&logo=book&logoColor=white">
    <img src="https://img.shields.io/badge/Google_Drive-34A853?style=for-the-badge&logo=googledrive&logoColor=white">
    <img src="https://img.shields.io/badge/Hugging_Face-FF5722?style=for-the-badge&logo=huggingface&logoColor=white">
    <img src="https://img.shields.io/badge/Caikit-008080?style=for-the-badge&logo=caikit&logoColor=white">
</div>

---

## 🎯 Overview
Proyek ini berfokus pada analisis sentimen menggunakan model **DistilBERT** dari Hugging Face yang diintegrasikan dengan **Caikit** untuk kemudahan pengembangan. Dengan model ini, kita dapat menganalisis apakah suatu teks bersentimen positif atau negatif. Contoh penerapan analisis sentimen meliputi ulasan produk, komentar media sosial, dan analisis opini publik.

## ⚙️ Requirements
- Python 3.6+
- Pustaka yang diperlukan: `transformers`, `torch`, dan `numpy`
  
Install pustaka yang diperlukan:
```bash
pip install transformers torch numpy
```

## 🚀 Model dan Tokenisasi
Model yang digunakan adalah DistilBERT (distilbert-base-uncased-finetuned-sst-2-english) dari pustaka Hugging Face. Model ini telah dilatih untuk tugas klasifikasi sentimen pada bahasa Inggris.

Tokenisasi Teks:

Teks input ditokenisasi dengan AutoTokenizer untuk mengubahnya menjadi bentuk yang dapat diproses oleh model BERT.
Token akan diubah ke dalam bentuk tensor (dalam format PyTorch) untuk memenuhi kebutuhan model.
Proses Prediksi:

Model menghasilkan keluaran dalam bentuk "logits" yang merepresentasikan kekuatan setiap kelas (positif atau negatif).
Softmax dan Probabilitas:

Keluaran "logits" diubah menjadi probabilitas menggunakan fungsi softmax. Probabilitas ini memberikan nilai keyakinan model pada setiap kelas.
Penentuan Sentimen:

Jika kelas 1 (positif) memiliki probabilitas tertinggi, maka hasilnya adalah positif. Jika tidak, hasilnya adalah negatif.
Confidence adalah nilai probabilitas tertinggi yang menunjukkan seberapa yakin model terhadap prediksinya.

## 📝 Kode Python
Berikut adalah kode untuk analisis sentimen menggunakan model ini:
```bash
# Import required libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load a sentiment analysis model from Hugging Face
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define a function for sentiment analysis
def analyze_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Get the model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_scores = probabilities.cpu().numpy()[0]  # Move to CPU before converting to NumPy

    # Determine sentiment
    sentiment = "positive" if np.argmax(sentiment_scores) == 1 else "negative"
    confidence = sentiment_scores[np.argmax(sentiment_scores)]

    return sentiment, confidence

# Test the function with a sample text
text = "I love playing MMORPG because it makes me feel so good!"
sentiment, confidence = analyze_sentiment(text)
print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")
```

## 📊 Contoh Analisis Sentimen
Berikut adalah dua contoh untuk memahami bagaimana model memprediksi sentimen dari suatu teks:

### Contoh 1

- Teks: "I love playing Ragnarok Online, but the probability drop item is frustrating!"
- Hasil: Sentimen Negatif dengan keyakinan 1.00
- Analisis: Meski kata "love" mengindikasikan aspek positif, model menangkap keluhan terkait "probability drop item" yang menggambarkan frustrasi terhadap game.


### Contoh 2

- Teks: "I enjoy playing MMORPGs because they make me feel great!"
- Hasil: Sentimen Positif dengan keyakinan 1.00
- Analisis: Frasa "make me feel great" memberikan indikasi positif yang kuat, sehingga model memprediksi sentimen positif.

## 📌 Kesimpulan
Model ini cukup andal dalam mendeteksi sentimen positif atau negatif, terutama pada kalimat yang memiliki kata atau frasa dengan makna kuat. Namun, pada teks yang ambigu atau rumit, hasil prediksi mungkin tidak sepenuhnya akurat, karena model ini hanya mengklasifikasikan teks ke dalam dua kelas (positif dan negatif) tanpa mempertimbangkan konteks yang lebih dalam.
