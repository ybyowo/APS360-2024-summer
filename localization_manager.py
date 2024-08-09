import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNNManager:
    def __init__(self, num_classes=2, model_path=None, device=None):
        """
        Initializes the Faster R-CNN model with specified number of classes and loads weights if provided.
        Args:
            num_classes (int): Number of classes for the dataset (including background).
            model_path (str, optional): Path to the model weights file.
            device (str, optional): The device to run the model on ('cuda' or 'cpu'). If None, it uses cuda if available.
        """
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(num_classes, model_path)

    def load_model(self, num_classes, model_path):
        """
        Loads or initializes the Faster R-CNN model.
        Args:
            num_classes (int): Number of classes.
            model_path (str, optional): Path to the custom model weights.
        
        Returns:
            model (torch.nn.Module): The loaded Faster R-CNN model.
        """
        # Load a model pre-trained on COCO
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the head with a new one (adjust number of classes)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model.to(self.device)
        return model

    def predict(self, images):
        """
        Make predictions on a list of images.
        Args:
            images (List[Tensor]): List of image tensors on which predictions need to be made.
        
        Returns:
            List[dict]: A list of dictionaries containing predictions.
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        return predictions

    def save_model(self, save_path):
        """
        Save the model weights to a file.
        Args:
            save_path (str): The path where to save the model.
        """
        torch.save(self.model.state_dict(), save_path)
