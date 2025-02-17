import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image

# Определение классов в соответствии с новой структурой dataset
skin_classes = {
    "appearance": ["healthy", "dull", "peeling", "redness", "grayish", "no_glow"],
    "texture": ["elastic", "tight", "smooth", "uneven", "dry"],
    "oiliness": ["no_oil", "oily_glow", "t_zone_oil", "evening_oil"],
    "problems/pores": ["small", "large", "blackheads", "clogged", "t_zone"],
    "problems/wrinkles": ["none", "fine_lines", "deep", "prone", "forehead", "eye_area", "nasolabial"],
    "problems/breakouts": ["pimples", "acne", "post_acne", "cheeks", "forehead", "chin", "jawline", "nose", "t_zone"],
    "goals": ["hydrate", "anti_aging", "tighten_pores", "reduce_redness", "smooth_texture", "control_oil", "brighten", "soothe", "reduce_wrinkles", "remove_pigmentation"],
    "race": [
        "nordic_caucasian", "central_caucasian", "mediterranean_caucasian",
        "east_asian", "southeast_asian", "south_asian",
        "light_african", "central_african", "deep_african",
        "light_latino", "mixed_latino", "deep_latino"
    ]
}

num_classes = sum(len(v) for v in skin_classes.values())

# Функция для сопоставления классов
class_labels = []
for category, subcategories in skin_classes.items():
    for subcategory in subcategories:
        class_labels.append(f"{category}/{subcategory}")

def get_class_index(class_name):
    return class_labels.index(class_name)

# Трансформации изображений для подготовки данных
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Кастомный датасет для multi-label классификации
class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for category, subcategories in skin_classes.items():
            for subcategory in subcategories:
                class_dir = os.path.join(root_dir, category, subcategory)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        label_vector = np.zeros(num_classes)  # One-Hot Encoding
                        label_vector[get_class_index(f"{category}/{subcategory}")] = 1
                        self.image_paths.append(img_path)
                        self.labels.append(label_vector)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Multi-label One-Hot Encoding
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Создание датасета
train_dataset = SkinDataset("dataset", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Создание модели
class SkinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SkinClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))  # Sigmoid for multi-label classification

model = SkinClassifier(num_classes)
criterion = nn.BCEWithLogitsLoss()  # Multi-label classification loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение модели

def train_model(num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  # BCEWithLogitsLoss handles multi-labels
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "skin_analyzer.pth")
    print("Модель сохранена в skin_analyzer.pth")

# Запуск обучения
# train_model()
