from ultralytics import YOLO
import torch

class YOLOManager:
    def __init__(self, model_path=None, device=None):
        """
        Initializes the YOLO model and loads weights if provided.
        Args:
            model_path (str, optional): Path to the model weights file.
            device (str, optional): The device to run the model on ('cuda' or 'cpu'). 
                                    If None, it uses cuda if available.
        """
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Loads the YOLO model.
        Args:
            model_path (str, optional): Path to the custom model weights.
        
        Returns:
            model (ultralytics.YOLO): The loaded YOLO model.
        """
        model = YOLO(model_path) if model_path else YOLO(pretrained=True)
        torch.cuda.set_device(0)
        return model

    def predict(self, images, size=640, conf_thres=0.5, iou_thres=0.3):
        """
        Make predictions on a list of images.
        Args:
            images (Tensor): Image tensor on which predictions need to be made.
            size (int): Size to which images are scaled.
            conf_thres (float): Confidence threshold.
            iou_thres (float): Intersection over union threshold.
        
        Returns:
            List[dict]: A list of dictionaries containing predictions.
        """
        results = self.model.predict(source=images, iou=iou_thres, conf=conf_thres, show_conf=False, show_labels=False)
        print(results[0])
        return results[0]

    def save_model(self, save_path):
        """
        Save the model weights to a file.
        Args:
            save_path (str): The path where to save the model.
        """
        torch.save(self.model.state_dict(), save_path)  # Changed to use PyTorch's save for state_dict
