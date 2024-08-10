import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw

def draw_box_on_image(image_path, box, save_path=None):
    """
    Draws a bounding box on an image and either saves it to a file or displays it using a high-quality method.
    
    Args:
        image_path (str): Path to the original image.
        box (list): A list containing the coordinates [xmin, ymin, xmax, ymax] of the box to draw.
        save_path (str, optional): If provided, the image will be saved to this path instead of displayed.
    """
    # Open the image file
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Draw the rectangle on the image
    draw.rectangle(box, outline='red', width=2)  # Adjust 'width' and 'outline' color as needed

    if save_path:
        # Save the modified image to file, specifying high quality
        img.save(save_path, 'JPEG', quality=99)
    else:
        # Show the image if no save path is provided
        img.show()