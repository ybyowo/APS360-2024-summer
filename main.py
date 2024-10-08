import os
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import pandas as pd
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision import models, transforms
from dataset_manager import LicensePlateDataset
from localization_manager import FasterRCNNManager
from util import draw_boxes_and_annotations
from ultralytics import YOLO
from PIL import Image
from segmentation_manager import YOLOManager
from eval import LicensePlateEvaluator

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
    # Initialize the YOLO model manager with the specified model and device
    yolo_manager = YOLOManager(segmentation_model_path, device)
    # Read the localization data from the CSV file
    df_localization = pd.read_csv(localization_csv_path)
    
    # List to store segmentation results
    segmentation_results = []
    # Set the target resize dimensions
    target_size = 640  # Fixed size to which images are resized for model input
    # Define the transformation to resize images and convert them to tensors
    transform = Compose([Resize((target_size, target_size)), ToTensor()])

    # Iterate over each entry in the localization CSV
    for index, row in df_localization.iterrows():
        img_filename = row['Image Filename'].strip("('").rstrip("',)")
        box = eval(row['Boxes'])

        # Construct the full path to the image
        img_path = os.path.join(images_folder, img_filename)
        img = Image.open(img_path).convert('RGB')
        original_width, original_height = img.size  # Original image dimensions

        # Crop the image according to the provided box coordinates
        img_cropped = img.crop((box[0], box[1], box[2], box[3]))
        cropped_width, cropped_height = img_cropped.size  # Dimensions of the cropped image

        # Transform the cropped image and prepare it for model prediction
        img_transformed = transform(img_cropped).unsqueeze(0).to(device)
        # Predict using the YOLO model
        results = yolo_manager.predict(img_transformed)
        
        # Process each result detected by the model
        for result in results:
            for bbox in result.boxes:
                # Scale the detected box coordinates back to the dimensions of the cropped image
                x1, y1, x2, y2 = bbox.xyxy[0].tolist()
                x1 = x1 * cropped_width / target_size + box[0]
                y1 = y1 * cropped_height / target_size + box[1]
                x2 = x2 * cropped_width / target_size + box[0]
                y2 = y2 * cropped_height / target_size + box[1]

                # Append the results with coordinates mapped back to the original image
                segmentation_results.append([img_filename, x1, y1, x2, y2])

    # Create a DataFrame from the segmentation results
    if segmentation_results:
        df_segmentation = pd.DataFrame(segmentation_results, columns=['Image Filename', 'x1', 'y1', 'x2', 'y2'])
        # Save the DataFrame to a CSV file
        df_segmentation.to_csv(save_csv_path, index=False)
        print(f"Segmentation results saved to {save_csv_path} with {len(df_segmentation)} records.")
    else:
        print("No segmentation results to save.")
        
def classify_characters_from_csv(csv_path, images_folder, model_path, save_results_path, device, num_classes=36):
    # Create an instance of the model architecture
    model = models.resnet101(weights=None)  # Initialize model without pretrained weights
    
    # Replace the fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load the model state dictionary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move model to the specified device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Read the CSV file containing segmentation data
    df = pd.read_csv(csv_path)
    classification_results = []

    # Process each row in the CSV
    for index, row in df.iterrows():
        img_filename = row['Image Filename']
        x1, y1, x2, y2 = map(float, [row['x1'], row['y1'], row['x2'], row['y2']])

        # Load and crop the image
        img_path = os.path.join(images_folder, img_filename)
        img = Image.open(img_path).convert('RGB')
        img_cropped = img.crop((x1, y1, x2, y2))

        # Transform the image for the model
        img_transformed = transform(img_cropped).unsqueeze(0).to(device)

        # Perform the prediction
        with torch.no_grad():
            outputs = model(img_transformed)
            _, predicted = outputs.max(1)
            predicted_class = predicted.item()

        # Append the result
        classification_results.append({
            'Image Filename': img_filename,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'Predicted Class': predicted_class
        })

    # Save the classification results to a new CSV
    df_results = pd.DataFrame(classification_results)
    df_results.to_csv(save_results_path, index=False)
    print(f"Classification results saved to {save_results_path}")
    
def visualize_predictions_from_csv(localization_csv, classification_csv, images_folder, output_folder):
    """
    Processes and visualizes predictions from localization and classification CSV files.
    
    Args:
        localization_csv (str): CSV file path containing localization predictions.
        classification_csv (str): CSV file path containing classification predictions.
        images_folder (str): Folder containing the images.
        output_folder (str): Folder to save the annotated images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the directory if it does not exist

    # Read the CSV files
    localization_df = pd.read_csv(localization_csv)
    classification_df = pd.read_csv(classification_csv)
    
    # Parse localization filenames
    localization_df['Image Filename'] = localization_df['Image Filename'].apply(lambda x: eval(x)[0])

    # Group by image filenames
    for filename in localization_df['Image Filename'].unique():
        loc_data = localization_df[localization_df['Image Filename'] == filename].to_dict('records')
        class_data = classification_df[classification_df['Image Filename'] == filename].to_dict('records')
        
        input_image_path = os.path.join(images_folder, filename)
        output_image_path = os.path.join(output_folder, f"annotated_{filename}")

        # Draw and save the image with annotations
        draw_boxes_and_annotations(input_image_path, loc_data, class_data, output_image_path)

    print(f"All images have been processed and saved in {output_folder}")
    
    
def main():
    # Configurations
    localization_weight = 'FastRCNN_learning_rate_0.001_batch_size_16_epoch_18.pth'
    localization_csv_path = 'localization.csv'
    segmentation_weight = "yolo_segmentation.pt"  
    segmentation_csv_path = 'segmentation.csv'
    classifier_weight = 'resnet_classifier.pth'
    classifier_csv_path = 'classification.csv'
    
    num_classes = 36  # Including background as one class
    contain_label = False
    dataset_path = 'test'
    images_folder = dataset_path  # Assuming images are in the same folder as the dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    output_folder = 'annotated_images'

    # Initialize dataset and DataLoader
    transform = ToTensor()
    dataset = LicensePlateDataset(root=dataset_path, transforms=transform, label=contain_label)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Perform localization
    perform_localization(data_loader, localization_weight, 2, localization_csv_path, device)
    # Perform segmentation from CSV
    perform_segmentation_from_csv(localization_csv_path, images_folder, segmentation_weight, segmentation_csv_path, device)
    # visualize_predictions_from_csv(segmentation_csv_path, images_folder, 'temp2')
    
    classify_characters_from_csv(segmentation_csv_path, images_folder, classifier_weight, classifier_csv_path, device, num_classes-1)
    visualize_predictions_from_csv(localization_csv_path, classifier_csv_path, images_folder, output_folder)
    
    # Evaluate if the dataset contains xml format label
    if contain_label == True:
        evaluator = LicensePlateEvaluator(localization_csv_path, classifier_csv_path, data_loader)
        results = evaluator.evaluate_predictions()
    
if __name__ == "__main__":
    main()
    