from pathlib import Path
import cv2
from yolo_wrapper import YoloWrapper

if __name__ == '__main__':
    # paths to the data
    dataset_path = Path('dataset')  # where the YOLO dataset will be
    large_field_images_path = Path('kaggle_larxel/images')  # where the original images
    # cropped_images_path = Path('data/raw_data/crops')
    xml_folder = Path('kaggle_larxel/annotations')
    
    labels_path = Path('kaggle_larxel/labels')  # where the labels are
    
    YoloWrapper.convert_xml_to_yolo(xml_folder, labels_path, 'licence')

    # create the dataset in the format of YOLO
    YoloWrapper.create_dataset(large_field_images_path, labels_path, dataset_path)
    # create YOLO configuration file
    config_path = 'kaggle_larxel_config.yaml'
    YoloWrapper.create_config_file(dataset_path, ['licence_plate'], config_path)

    # create pretrained YOLO model and train it using transfer learning
    model = YoloWrapper('nano')
    model.train(config_path, epochs=200, name='licence_plate')

    # make predictions on the validation set
    data_to_predict_path = dataset_path/'images'/'val'
    val_image_list = list(data_to_predict_path.glob('*.png'))

    # save the prediction in a csv file where the bounding boxes should have minimum size
    model.predict_and_save_to_csv(val_image_list, path_to_save_csv='nano_licence_plate.csv', minimum_size=100, threshold=0.25,
                                only_most_conf=True)
    # draw bounding boxes from csv
    for image in val_image_list:
        model.draw_bbox_from_csv(image, 'nano_license_plate.csv', image.stem)