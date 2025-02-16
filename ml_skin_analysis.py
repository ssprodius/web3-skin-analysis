import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import json

# Определение трансформаций для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Классификатор кожи с новыми характеристиками
class SkinAnalyzer(nn.Module):
    def __init__(self, num_classes=10):  # Увеличено число классов для разных типов кожи
        super(SkinAnalyzer, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Инициализация модели
model = SkinAnalyzer()
model.load_state_dict(torch.load("skin_analyzer.pth", map_location=torch.device('cpu')))
model.eval()

# Определение категорий
categories = {
    "oiliness": ["dry", "oily", "combo", "normal"],
    "pores": ["small", "large", "blackheads"],
    "breakouts": ["none", "forehead", "cheeks", "chin", "t_zone"],
    "wrinkles": ["none", "forehead", "eyes", "mouth"]
}

# Функция анализа кожи по изображению
def analyze_skin(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    
    # Определение категорий по пути к файлу
    detected_categories = []
    for category, subcategories in categories.items():
        for subcat in subcategories:
            if subcat in image_path:
                detected_categories.append(f"{category}: {subcat}")
    
    return {
        "predicted_skin_type": categories["oiliness"][predicted_class.item()],
        "detected_categories": detected_categories
    }

# Пример использования
# result = analyze_skin("dataset/oiliness/t_zone_oil/photo1.jpg")
# print("Результат анализа:", result)

