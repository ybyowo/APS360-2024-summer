import torchvision
import xmltodict
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import zipfile
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as T
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, Subset
import ssl
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from dataset import LicensePlateDataset
from utils import collate_fn, evaluate_model


ssl._create_default_https_context = ssl._create_unverified_context

transform = T.Compose([T.ToTensor()])
dataset = LicensePlateDataset('data/kaggle_larxel',  transforms=transform)

# Get dataset indices
dataset_size = len(dataset)
indices = list(range(dataset_size))

# Split indices into train, val, and test
train_indices, val_test_indices = train_test_split(indices, test_size=0.2, random_state=42)
val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=42)

# Create datasets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Create data loaders
train_loader_single = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader_single = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (license plate) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to the appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.00005)

# Training loop
lr = 0.0001
num_epochs = 40
i = 0
train_precisions = []
train_recalls = []
val_precisions = []
val_recalls = []
train_aps = []
val_aps = []
classification_losses = []
regression_losses = []
for epoch in range(num_epochs):
    for images, boxes, labels in train_loader:
        model.train()
        targets = []
        for box, label in zip(boxes, labels):
            target = {}
            target["boxes"] = box
            target["labels"] = label
            targets.append(target)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images = images.to(device)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())


        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Access individual losses
        classification_loss = loss_dict['loss_classifier']
        regression_loss = loss_dict['loss_box_reg']

        # Store losses
        classification_losses.append(float(classification_loss)/batch_size)
        regression_losses.append(float(regression_loss)/batch_size)

        print(f"Epoch: {epoch} Iteration: {i}")


        i += 1

        #lr_scheduler.step()
    train_recall, train_precision, train_ap = evaluate_model(model, train_loader_single, device)
    val_recall, val_precision, val_ap = evaluate_model(model, val_loader_single, device)

    train_recalls.append(train_recall)
    train_precisions.append(train_precision)
    val_recalls.append(val_recall)
    val_precisions.append(val_precision)
    train_aps.append(train_ap)
    val_aps.append(val_ap)
    # Save model checkpoint
    torch.save(model.state_dict(), f"FastRCNN_learning_rate_{lr}_batch_size_{batch_size}_epoch_{epoch}.pth")


# Plot the classification and regression losses
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(classification_losses, label='Classification Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Classification Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(regression_losses, label='Regression Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Regression Loss')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_recalls, label='Training Recall')
plt.plot(val_recalls, label='Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall Over Epochs')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_precisions, label='Training Precision')
plt.plot(val_precisions, label='Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision Over Epochs')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_aps, label='Training AP')
plt.plot(val_aps, label='Validation AP')
plt.xlabel('Epoch')
plt.ylabel('Average Precision')
plt.title('Average Precision Over Epochs')
plt.legend()
plt.show()

print("Training complete!")

