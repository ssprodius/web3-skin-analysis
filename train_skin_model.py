import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Определение структуры классов
CATEGORIES = {
    "appearance": ["healthy", "dull", "peeling", "redness", "grayish", "no_glow"],
    "texture": ["elastic", "tight", "smooth", "uneven", "dry"],
    "oiliness": ["no_oil", "oily_glow", "t_zone_oil", "evening_oil"],
    "problems/pores": ["small", "large", "blackheads", "clogged", "t_zone"],
    "problems/wrinkles": ["none", "fine_lines", "deep", "prone", "forehead", "eye_area", "nasolabial"],
    "problems/breakouts": ["pimples", "acne", "post_acne", "cheeks", "forehead", "chin", "jawline", "nose", "t_zone"],
    "goals": ["hydrate", "anti_aging", "tighten_pores", "reduce_redness", "smooth_texture", "control_oil", "brighten", "soothe", "reduce_wrinkles", "remove_pigmentation"]
}

NUM_CLASSES = sum(len(v) for v in CATEGORIES.values())

# Определение трансформаций
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка данных
train_dataset = datasets.ImageFolder(root="dataset", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Определение модели
class SkinClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SkinClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Инициализация модели, функции потерь и оптимизатора
model = SkinClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
def train_model(num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # Сохранение модели
    torch.save(model.state_dict(), "skin_classifier.pth")
    print("Model training complete and saved as skin_classifier.pth")

if __name__ == "__main__":
    train_model(10)

