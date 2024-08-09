import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def draw_box_on_image(image_path, box, save_path=None):
    """
    Draws a bounding box on an image and either displays it or saves it to a file.
    
    Args:
        image_path (str): Path to the original image.
        box (list): A list containing the coordinates [xmin, ymin, xmax, ymax] of the box to draw.
        save_path (str, optional): If provided, the image will be saved to this path instead of displayed.
    """
    # Open the image file
    img = Image.open(image_path)
    # Convert the image into a matplotlib format
    img = np.array(img, dtype=np.uint8)
    
    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # Calculate the width and height of the box
    width = box[2] - box[0]
    height = box[3] - box[1]
    
    # Create a rectangle patch
    rect = patches.Rectangle((box[0], box[1]), width, height, linewidth=2, edgecolor='r', facecolor='none')
    
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    if save_path:
        plt.savefig(save_path)  # Save the modified image to file
        plt.close(fig)  # Close the figure to free up memory
    else:
        plt.show()  # Display the image with the box