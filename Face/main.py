import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Face.model import CNNModel


# Класс для подготовки данных
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Загрузка и подготовка данных
data = pd.read_csv("data/fer2013.csv")
data['pixels'] = data['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape(48, 48, 1).astype(np.uint8))
images = np.stack(data['pixels'].values)
labels = data['emotion'].values

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = torch.as_tensor(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)

# Определение аугментаций для обучающего набора данных
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

# Для валидационного и тестового набора данных аугментации не применяются
val_test_transform = transforms.Compose([
    transforms.ToTensor()
])

# Создание наборов данных
train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
val_dataset = CustomDataset(X_val, y_val, transform=val_test_transform)
test_dataset = CustomDataset(X_test, y_test, transform=val_test_transform)

CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sad', 'Surprise', 'Neutral']

import seaborn as sns
import matplotlib.pyplot as plt

# Подсчет количества образцов каждого класса
unique, counts = np.unique(labels, return_counts=True)

# Создание DataFrame для удобства визуализации
data_for_plot = pd.DataFrame({'Class': unique, 'Counts': counts})

# Визуализация распределения классов
sns.barplot(x='Class', y='Counts', data=data_for_plot)
plt.title('Распределение классов в датасете')
plt.xlabel('Классы')
plt.ylabel('Количество образцов')
plt.xticks(range(len(unique)), CLASS_LABELS)  # Убедитесь, что CLASS_LABELS определены
plt.show()


# Создание загрузчиков данных (DataLoaders)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Определение модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Добавление списков для хранения истории обучения
train_losses = []
val_losses = []
val_accuracies = []

num_epochs = 100
for epoch in range(num_epochs):
    train_loss = 0
    val_loss = 0
    val_accuracy = 0
    total = 0

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss_temp = criterion(outputs, labels)
            val_loss += val_loss_temp.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            val_accuracy += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_accuracy = 100 * val_accuracy / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')



# import plotly.graph_objects as go
#
# # Создание графика для потерь
# loss_fig = go.Figure()
# loss_fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=train_losses, mode='lines+markers', name='Train Loss'))
# loss_fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=val_losses, mode='lines+markers', name='Validation Loss'))
# loss_fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
#
# # Показать график потерь
# loss_fig.show()
#
# # Создание графика для точности
# acc_fig = go.Figure()
# acc_fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=val_accuracies, mode='lines+markers', name='Validation Accuracy'))
# acc_fig.update_layout(title='Validation Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
#
# # Показать график точности
# acc_fig.show()


# Сохранение модели
torch.save(model.state_dict(), 'model100.pth')