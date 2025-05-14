import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader



data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
}

data_dir = r"C:\Users\yakup.erzengin\PycharmProjects\BulutProje\animals_split"

image_datasets = {
    x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
    for x in ['train', 'test']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True)
    for x in ['train', 'test']
}

class_names = image_datasets['train'].classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
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


model = SimpleCNN(num_classes=len(class_names)).to(device)




#ERKEN DURDURMA
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


#EĞİTİM FONKSİYONU

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=5):
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # ------- Eğitim ---------
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'])
        epoch_acc = correct.double() / len(image_datasets['train'])

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # ------- Validation --------
        val_loss = None
        if 'val' in dataloaders:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels.data)

            val_loss = val_running_loss / len(dataloaders['val'])
            val_acc = val_correct.double() / len(image_datasets['val'])

            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            early_stopping(val_loss)

            if early_stopping.early_stop:
                print("⛔️ Early stopping tetiklendi, eğitim durduruluyor.")
                break

        else:
            # Validation datası yoksa train loss'e göre erken durdurma yapar (çok sağlıklı değil ama opsiyon)
            early_stopping(epoch_loss)
            if early_stopping.early_stop:
                print("⛔️ Early stopping (train loss'e göre) tetiklendi, eğitim durduruluyor.")
                break

        print()  # boş satır

#MODEL EĞİTİMİ

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=5)


#Model Kaydetme

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'animals_full_checkpoint25.pth')


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# test et
evaluate_model(model, dataloaders['test'])
