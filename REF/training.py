import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Defina as classes de emoções
EMOTIONS = ["felicidade", "tristeza", "raiva", "neutro", "medo"]

# Crie uma classe personalizada para o conjunto de dados


class AudioEmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []  # Lista para armazenar pares (imagem, rótulo)

        # Carregar imagens e atribuir rótulos
        for i, emotion in enumerate(EMOTIONS):
            emotion_dir = os.path.join(root_dir, emotion)
            for filename in os.listdir(emotion_dir):
                image_path = os.path.join(emotion_dir, filename)
                self.data.append((image_path, i))  # (imagem, rótulo)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = plt.imread(image_path)  # Carregar imagem
        if self.transform:
            image = self.transform(image)
        return image, label


# Defina transformações de dados
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Crie conjuntos de treinamento e validação
train_dataset = AudioEmotionDataset(
    "REFT/REF/espectrogramas", transform=transform)

# Use train-test split para separar um subconjunto para validação
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size])

# Defina os dataloaders para treinamento e validação
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Crie o modelo ResNet-34 e defina a função de perda e otimizador
model = models.resnet34(pretrained=True)
num_features = model.fc.in_features
# Camada de saída com número de classes de emoções
model.fc = nn.Linear(num_features, len(EMOTIONS))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Treine o modelo


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        predicted_labels = []
        true_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

    return model, true_labels, predicted_labels


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, true_labels, predicted_labels = train_model(
    model, train_loader, val_loader, criterion, optimizer)

# Avalie o modelo
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=EMOTIONS))

# Matriz de confusão
confusion = confusion_matrix(true_labels, predicted_labels)
print("Matriz de Confusão:")
print(confusion)

# Salve o modelo treinado
torch.save(model.state_dict(
), "audio_emotion_model.pth")
