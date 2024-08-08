import torch
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Define data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load and prepare the model
def load_model(path, num_classes, device, pretrained=False):
    
    if pretrained:
        model = torch.load(path)
    else:
        model = models.resnet101(weights = 'DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model

# Option to use a pre-trained model
use_pretrained = False
model_path = 'model.pth' if use_pretrained else None

# Setup device
print("CUDA available:" , torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data_dir = './Letter_Dataset'
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(full_dataset.classes)

# Load model
model = load_model(model_path, num_classes, device, pretrained=use_pretrained)




# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Load data and split
targets = [sample[1] for sample in full_dataset.samples]
train_val_idx, test_idx = train_test_split(list(range(len(full_dataset))), test_size=0.15, stratify=targets)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1765, stratify=[targets[i] for i in train_val_idx])

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Functions for training and evaluation
def evaluate(model, loader):
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def save_model(model, path):
    # Save the model's state dictionary to the specified path
    torch.save(model.state_dict(), path)

# Training function with model saving logic
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.to(device)  
    best_accuracy = 0  # Variable to keep track of the best validation accuracy
    train_losses, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start time of the epoch
        
        model.train()  
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        epoch_duration = time.time() - epoch_start_time  # Calculate the duration of the epoch
        
        train_losses.append(train_loss / len(train_loader))
        val_accuracy = evaluate(model, val_loader)
        val_accuracies.append(val_accuracy)
        
        # Save the best performing model based on validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, 'best_model.pth')  # Save the best model state dict
            
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%, Duration: {epoch_duration:.2f} seconds')

    return train_losses, val_accuracies

# Train and evaluate
train_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss Trend')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy Trend')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()

# Final evaluation on test set
test_accuracy = evaluate(model, test_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')
