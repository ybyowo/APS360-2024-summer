from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from mapcalc import calculate_map


def collate_fn(batch):
    images = [item[0] for item in batch]
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # Find the max height and width in the batch
    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])

    # Pad images to the same size
    padded_images = []
    for img in images:
        padded_img = torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]), value=0)
        padded_images.append(padded_img)

    padded_images = torch.stack(padded_images, dim=0)
    return padded_images, boxes, labels

def evaluate_model(model, data_loader, device, iou_threshold=0.5, score_threshold=0.8):
    model.eval()
    correct = 0
    total_pred_boxes = 0
    total_true_boxes = 0
    ground_truth_dict = {"boxes": [], "labels": []}
    result_dict = {"boxes": [], "scores": [], "labels": []}

    with torch.no_grad():
        for images, boxes, labels in data_loader:
            images = [image.to(device) for image in images]
            targets = []
            for box, label in zip(boxes, labels):
                target = {}
                target["boxes"] = box
                target["labels"] = label
                targets.append(target)
                ground_truth_dict["boxes"].extend(box.cpu().numpy())
                ground_truth_dict["labels"].extend(label.cpu().numpy())
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                true_boxes = targets[i]['boxes']

                # Filter out predictions with low scores
                high_score_indices = pred_scores > score_threshold
                pred_boxes = pred_boxes[high_score_indices]
                pred_scores = pred_scores[high_score_indices]
                pred_labels = torch.ones_like(pred_scores)

                result_dict["boxes"].extend(pred_boxes.cpu().numpy())
                result_dict["scores"].extend(pred_scores.cpu().numpy())
                result_dict["labels"].extend(pred_labels.cpu().numpy())

                if pred_boxes.nelement() == 0 or true_boxes.nelement() == 0:
                    continue

                iou = torchvision.ops.box_iou(pred_boxes, true_boxes)
                max_iou, _ = iou.max(dim=1)
                correct += (max_iou > iou_threshold).sum().item()
                total_pred_boxes += pred_boxes.size(0)
                total_true_boxes += true_boxes.size(0)


    recall = correct / total_true_boxes if total_true_boxes > 0 else 0
    precision = correct / total_pred_boxes if total_pred_boxes > 0 else 0
    average_precision = calculate_map(ground_truth_dict, result_dict, iou_threshold)


    return recall, precision, average_precision



def visualize_image_with_boxes(img, target):
    fig, ax = plt.subplots(1)
    ax.imshow(img.permute(1, 2, 0).cpu().detach().numpy()) # Convert from (C, H, W) to (H, W, C) for matplotlib

    for i, box in enumerate(target['boxes']):
        if target['scores'][i] > 0.5:
          rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
          ax.add_patch(rect)

    plt.show()
