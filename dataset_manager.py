import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset

class LicensePlateDataset(Dataset):
    def __init__(self, root, transforms=None):
        """
        Initializes the dataset.
        Args:
            root (str): The root directory of the dataset containing image files and corresponding XML annotation files.
            transforms (callable, optional): A function/transform that takes in an image and returns a transformed version.
        """
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted([file for file in os.listdir(root) if file.endswith('.jpg')]))

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            tuple: Tuple containing the image, bounding boxes, labels for the objects in the image, and the image file name.
        """
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')

        # Extract the image file name without the path
        image_filename = os.path.basename(img_path)

        ann_path = img_path.replace('.jpg', '.xml')
        boxes, labels = self.parse_annotations(ann_path)

        if self.transforms:
            img = self.transforms(img)

        # Include the image filename in the return tuple
        return img, boxes, labels, image_filename



    def parse_annotations(self, ann_path):
        """
        Parses the XML annotation file.
        Args:
            ann_path (str): Path to the XML annotation file.
        
        Returns:
            tuple: A tuple containing all bounding boxes and labels extracted from the annotation file.
        """
        tree = ET.parse(ann_path)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall('object'):
            label = obj.find('name').text
            label_id = self.label_to_int(label)

            if label_id == -1:
                continue  # Skip unknown labels

            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return boxes, labels

    def label_to_int(self, label):
        """
        Converts label names to integer IDs.
        Args:
            label (str): The name of the label.
        
        Returns:
            int: The integer label ID, or -1 if the label is unknown.
        """
        label_map = {
            'Plate': 0,
            '0': 1, '1': 2, '2': 3, '3': 4, '4': 5,
            '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
            'A': 11, 'B': 12, 'C': 13, 'D': 14, 'E': 15,
            'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
            'K': 21, 'L': 22, 'M': 23, 'N': 24, 'P': 25,
            'Q': 26, 'R': 27, 'S': 28, 'T': 29,
            'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
            'Z': 35
        }
        return label_map.get(label, -1)  # Returns -1 if the label is not found

    def __len__(self):
        """
        Returns the number of items in the dataset.
        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.imgs)

