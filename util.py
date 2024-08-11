import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def map_class_id_to_label(class_id):
    """
    Maps class ID to the corresponding label, skipping 'O'.
    
    Args:
        class_id (int): Class ID from 0-34, skipping 'O' which would be 24.
    
    Returns:
        str: Corresponding label.
    """
    if 0 <= class_id <= 9:
        return str(class_id)  # Return numbers as strings
    elif 10 <= class_id <= 23:
        return chr(class_id + 55)  # ASCII 'A' is 65, so A corresponds to 10, B to 11, ..., I to 18
    elif 24 <= class_id:
        return chr(class_id + 56)  # Skip 'O', so J corresponds to 19, ..., Z to 35 (should be 34 with the skip)
 
def calculate_font_size(image_width):
    """
    Calculate an appropriate font size based on the width of the image.
    For example, set the font size to be 1/40 of the image width.
    """
    return max(24, int(image_width / 40))  # Ensure the font isn't too small, adjust ratio as needed
    
def draw_boxes_and_annotations(image_path, localization_boxes, classification_data, save_path):
    """
    Draws bounding boxes on an image for localization and classification, with annotations.
    
    Args:
        image_path (str): Path to the original image.
        localization_boxes (list): List of dictionaries with localization data.
        classification_data (list): List of dictionaries with classification results.
        save_path (str): Path to save the annotated image.
    """
    # Open the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font_path = 'arial.ttf'
    font_size_1 = calculate_font_size(img.width * 0.5)
    font_1 = ImageFont.truetype(font_path, font_size_1)  
    font_size_2 = calculate_font_size(img.width)
    font_2 = ImageFont.truetype(font_path, font_size_2)
    for data in localization_boxes:
        box = eval(data['Boxes'])
        score = data['Scores']
        draw.rectangle(box, outline='red', width=5)
        draw.text((box[0], box[1] - 10), f'Score {score:.2f}', fill='yellow', font=font_1)

    # Draw classification boxes and annotations
    for data in classification_data:
        box = [data['x1'], data['y1'], data['x2'], data['y2']]
        predicted_class = map_class_id_to_label(data['Predicted Class'])
        draw.rectangle(box, outline='blue', width=3)
        text_pos = (max(box[0], 0), min(box[3] + 5, img.height - 20))
        draw.text(text_pos, predicted_class, fill='red', font=font_2)

    # Save the annotated image
    img.save(save_path, 'JPEG', quality=99)
