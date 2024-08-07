import torch, torchvision
from utils import visualize_images_with_boxes, evaluate_model
from dataset import LicensePlateDataset
from torch.utils.data import DataLoader, ConcatDataset

# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the box_predictor with a new one
num_classes = 2  # 1 class (license plate) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the best model weights from training
model.load_state_dict(torch.load('best_model', map_location=device))

# Move model to the appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load the saved datasets and create loaders
train_set = torch.load('train_dataset')
val_set = torch.load('val_dataset')
test_set = torch.load('test_dataset')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Create a dataset and loader for self-taken images and combine with the test set
new_dataset = LicensePlateDataset('lp_loc',  transforms=transform)
new_dataset_loader = DataLoader(new_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
combined_test_set = ConcatDataset([test_dataset, new_dataset])
combined_test_loader = DataLoader(combined_test_set, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Recall, precision, and average precision for the best model on the training set
train_recall, train_precision, train_ap = evaluate_model(model, train_loader, device)
print(f"Training Set - Recall: {train_recall}, Precision: {train_precision}, Average Precision: {train_ap}")

# Recall, precision, and average precision for the best model on the validation set
val_recall, val_precision, val_ap = evaluate_model(model, val_loader, device)
print(f"Validation Set - Recall: {val_recall}, Precision: {val_precision}, Average Precision: {val_ap}")

# Recall, precision, and average precision for the best model on the combined test set
combined_recall, combined_precision, combined_ap = evaluate_model(model, combined_test_loader, device)
print(f"Combined Test Set - Recall: {combined_recall}, Precision: {combined_precision}, Average Precision: {combined_ap}")

# Visualize the bounding box predictions for the combined test set
model.eval()
model = model.to('cpu')
for images, boxes, labels in combined_test_loader:
    with torch.no_grad():
        targets = []
        for box, label in zip(boxes, labels):
            target = {}
            target["boxes"] = box
            target["labels"] = label
            targets.append(target)
        images = images[:,:3,:,:]
        pred = model(images)

        visualize_image_with_boxes(images[0], pred[0])
