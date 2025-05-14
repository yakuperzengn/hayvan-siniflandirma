import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

# ✅ Sınıf isimleri (10 sınıf)
class_names = ['Kelebek', 'Kedi', 'Tavuk', 'İnek', 'Köpek', 'Fil', 'At', 'Koyun', 'Örümcek', 'Sincap']

# ✅ Model tanımı (128x128 giriş, 10 sınıf)
class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ✅ Modeli yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCNN().to(device)
checkpoint = torch.load("model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ✅ Görseli hazırla
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ✅ Tahmin fonksiyonu
def predict(image):
    image_rgb = image.convert("RGB")
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_index]
        confidence = probabilities[predicted_index].item()

    label = f"🐾 Tahmin: {predicted_class} (%{confidence * 100:.2f})"
    return image, label

# ✅ Geliştirilmiş Gradio Arayüz
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Bir hayvan görseli yükle"),
    outputs=[
        gr.Image(type="pil", label="Yüklenen Görsel"),
        gr.Label(label="Tahmin Sonucu")
    ],
    title="Hayvan Sınıflandırıcı",
    description="Resim yükle, CNN modeliyle hangi hayvan olduğunu tahmin edelim!",
    allow_flagging="never",
    theme="default"
).launch()
