from PIL import Image
import os

# Path to the folder containing images
folder_path = '/content/drive/My Drive/kaggle_larxel/images'

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

# Resize images in the specified folder
resize_images(folder_path)