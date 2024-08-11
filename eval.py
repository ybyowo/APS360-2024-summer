import pandas as pd
import torch

class LicensePlateEvaluator:
    def __init__(self, localization_csv, character_csv, data_loader, iou_threshold=0.5):
        self.localization_data = self.parse_csv(localization_csv, is_localization=True)
        self.character_data = self.parse_csv(character_csv, is_localization=False)
        self.data_loader = data_loader
        self.iou_threshold = iou_threshold

    def bbox_iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - inter_area
        return inter_area / union if union != 0 else 0
    
    def parse_csv(self, file_path, is_localization=False):
        data = pd.read_csv(file_path)
        if is_localization:
            # Handle localization data with tuple-like filenames and list-like Boxes
            data['Image Filename'] = data['Image Filename'].apply(lambda x: eval(x)[0])
            data['Boxes'] = data['Boxes'].apply(eval)  # Convert string representation of lists to actual lists
        else:
            # Handle character classification data with separate coordinate columns
            if {'x1', 'y1', 'x2', 'y2', 'Predicted Class'}.issubset(data.columns):
                data['Boxes'] = data.apply(lambda row: [row['x1'], row['y1'], row['x2'], row['y2']], axis=1)
            else:
                raise ValueError("Character classification CSV is missing required columns.")
        return data

    def translate_label(self, pred_class):
        mapping = {
            i: i + 1 for i in range(10)  # Digits (0-9 to 1-10)
        }
        offset = 10
        # Add mappings for letters, skipping 'O'
        for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            if char != 'O':
                mapping[char] = offset + (i if char < 'O' else i - 1)
        return mapping.get(pred_class, -1)

    def evaluate_predictions(self):
        # Metrics for localization
        tp_plate, fp_plate, fn_plate = 0, 0, 0
        # Metrics for segmentation (character bounding boxes)
        tp_seg, fp_seg, fn_seg = 0, 0, 0
        # Metrics for classification (character classes)
        tp_class, fp_class, fn_class = 0, 0, 0

        for images, gt_boxes, gt_labels, img_filenames in self.data_loader:
            for img, gt_box, gt_label, img_filename in zip(images, gt_boxes, gt_labels, img_filenames):
                # Localize license plates
                pred_boxes_plate = self.localization_data[self.localization_data['Image Filename'] == img_filename]['Boxes'].tolist()
                matched_plate = self.match_predictions(gt_box, gt_label, pred_boxes_plate, [0]*len(pred_boxes_plate), 0)
                tp_plate += matched_plate['tp']
                fp_plate += matched_plate['fp']
                fn_plate += matched_plate['fn']

                # Segment and classify characters
                pred_boxes_char = self.character_data[self.character_data['Image Filename'] == img_filename]['Boxes'].tolist()
                pred_labels_char = [self.translate_label(label) for label in self.character_data[self.character_data['Image Filename'] == img_filename]['Predicted Class']]
                
                # Match segmentation
                matched_seg = self.match_predictions(gt_box, gt_label, pred_boxes_char, [1]*len(pred_boxes_char), 1)
                tp_seg += matched_seg['tp']
                fp_seg += matched_seg['fp']
                fn_seg += matched_seg['fn']

                # Match classification
                matched_class = self.match_classification(gt_box, gt_label, pred_boxes_char, pred_labels_char, 1)
                tp_class += matched_class['tp']
                fp_class += matched_class['fp']
                fn_class += matched_class['fn']

        # Print and return results for all tasks
        results_plate = self.print_evaluation_results(tp_plate, fp_plate, fn_plate, "License Plate Localization")
        results_seg = self.print_evaluation_results(tp_seg, fp_seg, fn_seg, "Character Segmentation")
        results_class = self.print_evaluation_results(tp_class, fp_class, fn_class, "Character Classification")
        return {"license_plate": results_plate, "segmentation": results_seg, "classification": results_class}

    def match_predictions(self, gt_boxes, gt_labels, pred_boxes, pred_labels, label):
        tp, fp, fn = 0, 0, 0
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            if pred_label == label:
                match_found = False
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    if self.bbox_iou(pred_box, gt_box) >= self.iou_threshold and gt_label == label:
                        tp += 1
                        match_found = True
                        break
                if not match_found:
                    fp += 1
        fn = len([1 for gt_label in gt_labels if gt_label == label]) - tp
        return {'tp': tp, 'fp': fp, 'fn': fn}
    
    def match_classification(self, gt_boxes, gt_labels, pred_boxes, pred_labels, label):
        # This method assumes that classification accuracy is judged by correct label prediction for correctly detected characters
        tp, fp, fn = 0, 0, 0
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            match_found = False
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                if self.bbox_iou(pred_box, gt_box) >= self.iou_threshold and gt_label == label:
                    if pred_label == gt_label:
                        tp += 1
                    else:
                        fp += 1
                    match_found = True
                    break
            if not match_found:
                fp += 1
        fn = len([1 for gt_label in gt_labels if gt_label == label]) - tp
        return {'tp': tp, 'fp': fp, 'fn': fn}
    
    def print_evaluation_results(self, tp, fp, fn, task_name):
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        results = {"precision": precision, "recall": recall, "f1_score": f1_score}
        print(f"{task_name} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
        return results


