import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from dataset import LicensePlateDataset
import pandas as pd
import argparse
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
parser.add_argument('--score-threshold', type=float, default=0.8, help='Score threshold for bounding box predictions')
args = parser.parse_args()

# Set up data transformations
transform = T.Compose([T.ToTensor()])
dataset = LicensePlateDataset(args.dataset_path, transforms=transform)

# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2  # 1 class (license plate) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load trained model weights
model.load_state_dict(torch.load(args.model_path, map_location=device))

# Perform forward pass and save predictions
model.eval()
predictions = {}

for idx in range(len(dataset)):
    image, _, _= dataset[idx]
    image_filename = dataset.imgs[idx]
    image = image.unsqueeze(0).to(device)
    outputs = model(image)

    boxes_scores = []
    for box, score in zip(outputs[0]['boxes'], outputs[0]['scores']):
        if score > args.score_threshold:
            # Save bounding boxes in [xmin, ymin, xmax, ymax] format
            boxes_scores.append([box[0].item(), box[1].item(), box[2].item(), box[3].item()])
    
    if boxes_scores:
        predictions[image_filename] = boxes_scores

# Convert predictions to DataFrame
df = pd.DataFrame(list(predictions.items()), columns=['image_filename', 'boxes'])

# Save DataFrame to a CSV file
df.to_csv('predicted_bounding_boxes.csv', index=False)

print("Predictions saved to predicted_bounding_boxes.csv")
