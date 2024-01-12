import torch
import torch.nn as nn



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2)
        self.conv4 = nn.Conv2d(128, 512, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 8)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.batchnorm1(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.batchnorm2(torch.relu(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.batchnorm3(torch.relu(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)  # После этой строки данные "выравниваются" в одномерный вектор
        x = torch.relu(self.fc1(x))  # Тут важно, чтобы размер данных соответствовал входу self.fc1
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x