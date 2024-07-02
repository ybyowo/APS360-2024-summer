from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision


# Function to resize images to the smallest image size
def resize_images(folder_path):
    min_width, min_height = float('inf'), float('inf')
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            width, height = img.size
            if width < min_width:
                min_width = width
            if height < min_height:
                min_height = height
    # Resize all images to the smallest size found
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            img_resized = img.resize((min_width, min_height), Image.ANTIALIAS)
            # Save or overwrite the resized image
            img_resized.save(image_path)

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
    plt.xlabel('Epoch')
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
                print(pred_boxes)
                print(true_boxes)

                if pred_boxes.nelement() == 0 or true_boxes.nelement() == 0:
                    continue

                iou = torchvision.ops.box_iou(pred_boxes, true_boxes)
                print(f"iou: {iou}")
                max_iou, _ = iou.max(dim=1)
                correct += (max_iou > 0.5).sum().item()
                total += pred_boxes.size(0)

    return correct / total if total > 0 else 0



def visualize_image_with_boxes(img, target):
    fig, ax = plt.subplots(1)
    ax.imshow(img.permute(1, 2, 0).cpu().detach().numpy()) # Convert from (C, H, W) to (H, W, C) for matplotlib

    for i, box in enumerate(target['boxes']):
        if target['scores'][i] > 0.5:
          rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
          ax.add_patch(rect)

    plt.show()