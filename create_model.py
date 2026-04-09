"""
Script to create and save a KL-grading model based on ResNet18.
This creates a model pretrained on ImageNet, with the final layer
adapted for 5 classes (KL Grade 0-4).

Run this once: py create_model.py
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

print("Creating KL-grading model based on ResNet18...")

# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# --- Adapt for grayscale X-ray images ---
# X-rays are grayscale but we can use the RGB model by converting to 3-channel
# or replace first conv layer to accept 1 channel
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Replace final FC layer for 5 classes (Grade 0, 1, 2, 3, 4)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 5)
)

# Save the model
model_path = Path(__file__).parent / "model.pth"
torch.save(model.state_dict(), str(model_path))
print(f"Model saved to: {model_path}")
print(f"Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
print("\nNOTE: This model uses ImageNet pretrained weights for feature extraction.")
print("For accurate clinical results, fine-tune on labeled knee X-ray dataset.")
print("Example: OAI (Osteoarthritis Initiative) dataset from NCBI.")
