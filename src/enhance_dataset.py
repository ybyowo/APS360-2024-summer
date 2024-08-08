import os
from PIL import Image, ImageEnhance
import random
import shutil
import argparse

def darken_images(dataset_path, darken_prob=0.5):
    # Determine the parent directory and create the new directory for the enhanced dataset
    parent_dir = os.path.dirname(dataset_path)
    output_path = os.path.join(parent_dir, 'enhanced_dataset')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'annotations'), exist_ok=True)

    for filename in os.listdir(os.path.join(dataset_path, 'images')):
        img_path = os.path.join(dataset_path, 'images', filename)
        img = Image.open(img_path).convert("RGB")

        # Save the original image and annotation
        shutil.copy(img_path, os.path.join(output_path, 'images', filename))
        shutil.copy(os.path.join(dataset_path, 'annotations', os.path.splitext(filename)[0] + '.xml'),
                    os.path.join(output_path, 'annotations', os.path.splitext(filename)[0] + '.xml'))

        # Apply darkening with a certain probability and save the darkened image
        if random.random() < darken_prob:
            enhancer = ImageEnhance.Brightness(img)
            dark_img = enhancer.enhance(0.1)
            dark_img_filename = 'dark_' + filename
            dark_img.save(os.path.join(output_path, 'images', dark_img_filename))

            # Copy the annotation for the darkened image
            shutil.copy(os.path.join(dataset_path, 'annotations', os.path.splitext(filename)[0] + '.xml'),
                        os.path.join(output_path, 'annotations', os.path.splitext(dark_img_filename)[0] + '.xml'))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--darken-prob', type=float, default=0.5, help='Probability of darkening an image')
args = parser.parse_args()

# Enhance dataset with darkened images
darken_images(args.dataset_path, args.darken_prob)
print(f"Enhanced dataset with darkened images saved to {os.path.join(os.path.dirname(args.dataset_path), 'enhanced_dataset')}")
