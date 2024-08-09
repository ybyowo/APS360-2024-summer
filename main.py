import os
import torch
from torchvision.transforms import ToTensor
import pandas as pd
from torchvision.transforms import Compose, Resize, ToTensor
from dataset_manager import LicensePlateDataset
from localization_manager import FasterRCNNManager
from util import draw_box_on_image
from ultralytics import YOLO
from PIL import Image
from segmentation_manager import YOLOManager

def perform_localization(data_loader, model_path, num_classes, save_csv_path, device):
    # Initialize model
    faster_rcnn = FasterRCNNManager(num_classes=2, model_path=model_path, device=device)

    # Make predictions
    predictions = []
    for images, boxes, labels, image_filenames in data_loader:  # Now includes image_filenames
        images = images.to(device)
        images = images.squeeze(0)  # Ensure the input tensor is correctly formatted.
        output = faster_rcnn.predict([images])
        for pred in output:
            if pred['scores'].nelement() == 0:
                # No detections, optionally continue or record a default prediction
                continue
            max_score_idx = pred['scores'].argmax()  # Index of the highest scoring prediction
            max_box = pred['boxes'][max_score_idx].tolist()
            max_score = pred['scores'][max_score_idx].item()
            max_label = pred['labels'][max_score_idx].item()
            predictions.append((image_filenames, max_box, max_score, max_label))
            
    # Process predictions into DataFrame
    df = pd.DataFrame(predictions, columns=['Image Filename', 'Boxes', 'Scores', 'Labels'])

    # Save to CSV
    df.to_csv(save_csv_path, index=False)
    print(f"Predictions saved to {save_csv_path}")
    
def perform_segmentation_from_csv(localization_csv_path, images_folder, segmentation_model_path, save_csv_path, device):
    yolo_manager = YOLOManager(segmentation_model_path, device)
    df_localization = pd.read_csv(localization_csv_path)
    
    segmentation_results = []
    transform = Compose([Resize((640, 640)), ToTensor()])
    
    for index, row in df_localization.iterrows():
        img_filename = row['Image Filename'].strip("('").rstrip("',)")
        box = eval(row['Boxes'])

        img_path = os.path.join(images_folder, img_filename)
        img = Image.open(img_path).convert('RGB')
        img_cropped = img.crop((box[0], box[1], box[2], box[3]))
        img_transformed = img_transformed = transform(img_cropped).unsqueeze(0).to(device)
        results = yolo_manager.predict(img_transformed)
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = bbox.xyxy[0].tolist()
                # x1, y1, x2, y2, conf, cls_id = bbox
                segmentation_results.append([img_filename, x1, y1, x2, y2])

    df_segmentation = pd.DataFrame(segmentation_results, columns=['Image Filename', 'x1', 'y1', 'x2', 'y2'])
    df_segmentation.to_csv(save_csv_path, index=False)
    print(f"Segmentation results saved to {save_csv_path}")



def visualize_predictions_from_csv(csv_path, images_folder, output_folder):
    """
    Reads predictions from a CSV file and draws bounding boxes on the corresponding images.
    
    Args:
        csv_path (str): Path to the CSV file containing predictions.
        images_folder (str): Path to the folder containing the images.
        output_folder (str): Path to the folder where annotated images will be saved.
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Use os.makedirs to create the directory if it does not exist

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # Parse image filename, removing extra characters from tuple-string format
        image_filename = eval(row['Image Filename'])[0]  # Safe because we know the content structure
        box = eval(row['Boxes'])  # Ensure that the box coordinates are in list format
        
        # Define the path to the input image and the output image
        input_image_path = os.path.join(images_folder, image_filename)
        output_image_path = os.path.join(output_folder, f"annotated_{image_filename}")
        
        # Draw and save the image with the box
        draw_box_on_image(input_image_path, box, output_image_path)

    print(f"All images have been processed and saved in {output_folder}")
    
    
def main():
    # Configurations
    localization_weight = 'FastRCNN_learning_rate_0.001_batch_size_16_epoch_18.pth'
    localization_csv_path = 'localization.csv'
    segmentation_weight = "yolo_segmentation.pt"  
    segmentation_csv_path = 'segmentation.csv'
    
    num_classes = 37  # Including background as one class
    dataset_path = 'C:\\Users\\Wang Yanchi\\Desktop\\uoft\\3\\aps360\\ALPR\\ALPR_repo\\APS360-2024-summer2\\APS360-2024-summer\\test'
    images_folder = dataset_path  # Assuming images are in the same folder as the dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    output_folder = 'annotated_images'

    # Initialize dataset and DataLoader
    transform = ToTensor()
    dataset = LicensePlateDataset(root=dataset_path, transforms=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Perform localization
    perform_localization(data_loader, localization_weight, 2, localization_csv_path, device)
    visualize_predictions_from_csv(localization_csv_path, images_folder, 'temp')
    # Perform segmentation from CSV
    perform_segmentation_from_csv(localization_csv_path, images_folder, segmentation_weight, segmentation_csv_path, device)
    # visualize_predictions_from_csv(segmentation_csv_path, images_folder, 'temp2')

if __name__ == "__main__":
    main()
    