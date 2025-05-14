
# 🧠 Hayvan Sınıflandırma Projesi (CNN + PyTorch)

Bu proje, hayvan resimlerini sınıflandırmak için PyTorch ile eğitilmiş bir Convolutional Neural Network (CNN) modelini içerir. Eğitim Google Colab ortamında yapılmış, ardından model `.pt` formatında kaydedilip Gradio üzerinden kullanıcı dostu bir arayüzle sunulmuştur.

## 📁 Proje Yapısı

```
📦 Hayvan-Siniflandirma-Projesi
├── animals_model.pt           # Eğitilmiş model ,Drive Linki Yazıldı
├── app.py                     # Gradio tabanlı görsel sınıflandırma arayüzü
├── README.md                  # Bu döküman
├── requirements.txt           # Gerekli Python kütüphaneleri
└── dataset/                   # Eğitimde kullanılan veri seti (Animals-10)
```
## MODEL
-https://drive.google.com/file/d/10ES5Hs1GJ5K_mai8O2ur5mzzWLSYxwwi/view?usp=sharing


## 🔧 Kullanılan Teknolojiler

- Python 🐍
- PyTorch 🔥
- Google Colab
- Gradio 🌐 (veya PyCharm arayüzü)
- Matplotlib 📊
- sklearn 🎯

## 🚀 Eğitim (Colab Üzerinden)

```python
# Google Colab ortamında:
!pip install torch torchvision matplotlib
!git clone https://github.com/yourgithub/yourrepo.git
```

Eğitim sırasında:
- Loss & Accuracy grafikleri çizildi
- F1 Score hesaplandı
- Model `.pt` olarak kaydedildi

## 🖼️ Arayüz (Gradio)

Kullanıcı bir resim yükler → model tahmin yapar → sonucu gösterir.

```bash
python app.py
```

Uygulama gibi tarayıcıda çalışır.

## 🔍 Sınıflar

```
['At', 'Fil', 'Inek', 'Kedi', 'Kelebek', 'Kopek', 'Koyun', 'Orumcek', 'Sincap', 'Tavuk']
```

## ✅ Eğitim Özeti

- Model: SimpleCNN (2 Conv, 2 Dense)
- Doğruluk (Accuracy): %92
- F1 Score: 0.89 (weighted avg)
- Epoch: 25

## 📌 Proje Sahibi

> 👤 Yakup Erzengin  
> 📍 Erzurum,Türkiye - yakuperzengin0@gmail.com
> 🎓 Bulut Bilişim ve Yapay Zeka Dersi  
> 🧪 2025

## 📎 Notlar

- Veri seti: [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- Eğitim/deneme/test ayrımı yapılmıştır
- CNN modeli kullanıcıdan gelen resmi değerlendirerek sınıf tahmini yapar

✨ Bu projeyle hem görüntü işleme hem de makine öğrenimi uygulaması geliştirildi. Kodlar PyTorch ile yazıldı, arayüz Gradio ile desteklendi.
