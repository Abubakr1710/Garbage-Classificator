import torch
from torchvision.models import resnext50_32x4d
import torch.nn as nn


model = resnext50_32x4d(pretrained=True)
inputs = model.fc.in_features
output = 6
clf = nn.Linear(inputs,output)
model.fc = clf