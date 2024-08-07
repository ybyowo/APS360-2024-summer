import torchvision
import xmltodict
from PIL import Image
import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class LicensePlateDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        ann_path = os.path.join(self.root, "annotations", self.labels[idx])
        img = Image.open(img_path)

        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        if self.transforms:
            img = self.transforms(img)

        img = img[:3, :, :]
        return img, boxes, labels

    def __len__(self):
        return len(self.imgs)
