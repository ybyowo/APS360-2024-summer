import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_image_with_boxes(img, target):
    fig, ax = plt.subplots(1)
    ax.imshow(img.permute(1, 2, 0).cpu().detach().numpy()) # Convert from (C, H, W) to (H, W, C) for matplotlib

    for i, box in enumerate(target['boxes']):
        if target['scores'][i] > 0.5:
          rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
          ax.add_patch(rect)

    plt.show()


model.eval()
model.to('cpu')
for images, boxes, labels in data_loader:
    with torch.no_grad():
        targets = []
        for box, label in zip(boxes, labels):
            target = {}
            target["boxes"] = box.squeeze(0)
            target["labels"] = label
            targets.append(target)

        #evaluate_predictions(pred, targets)
        #visualize_image_with_boxes(images[0], pred[0])

