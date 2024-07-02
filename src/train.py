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

ssl._create_default_https_context = ssl._create_unverified_context


def collate_fn(batch):
    images = [item[0] for item in batch]
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    images = torch.stack(images, dim=0)
    
    return images, boxes, labels

def plot_training_validation_accuracy(train_acc, val_acc):
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, boxes, labels in data_loader:
            images = list(image.to(device) for image in images)
            targets = []
            for box, label in zip(boxes, labels):
                target = {}
                target["boxes"] = box
                target["labels"] = label
                targets.append(target)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes']
                true_boxes = targets[i]['boxes']

                if pred_boxes.nelement() == 0 or true_boxes.nelement() == 0:
                    continue

                iou = torchvision.ops.box_iou(pred_boxes, true_boxes)
                max_iou, _ = iou.max(dim=1)
                correct += (max_iou > 0.5).sum().item()
                total += true_boxes.size(0)

    return correct / total if total > 0 else 0

class LicensePlateDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        ann_path = os.path.join(self.root, "annotations", self.labels[idx])
        img = Image.open(img_path)

        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        if self.transforms:
            img = self.transforms(img)

        img, boxes = resize_image(img, boxes)
        img = img[:3, :, :]
        return img, boxes, labels

    def __len__(self):
        return len(self.imgs)

transform = T.Compose([T.ToTensor()])
dataset = LicensePlateDataset('../data/kaggle_larxel',  transforms=transform)
data_loader = DataLoader(dataset, shuffle=True, batch_size=4)

# Get dataset indices
dataset_size = len(dataset)
indices = list(range(dataset_size))

# Split indices into train, val, and test
train_indices, val_test_indices = train_test_split(indices, test_size=0.2, random_state=42)
val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=42)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


batch_size = 4

# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (license plate) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to the appropriate device
device = 'cpu'
model.to(device)

# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 1
i = 0
train_accuracies = []
val_accuracies = []
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
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Access individual losses
        classification_loss = loss_dict['loss_classifier']
        regression_loss = loss_dict['loss_box_reg']

        # Store losses
        classification_losses.append(float(classification_loss)/batch_size)
        regression_losses.append(float(regression_loss)/batch_size)
        
        if i % 10 == 0:
            train_accuracy = evaluate_model(model, train_loader, device)
            val_accuracy = evaluate_model(model, val_loader, device)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

        i += 1

        lr_scheduler.step()

torch.save(model.state_dict(), f'Epochs_{num_epochs}_batch_size_{batch_size}_lr_{lr}.pth')

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

plot_training_validation_accuracy(train_accuracies, val_accuracies)

print("Training complete!")

