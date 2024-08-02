from ultralytics import YOLO
import torch
from PIL import Image

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def print_final_answer(results):
    sorted_tensor = 0
    l=[]
    for i, r in enumerate(results):
        # Save results to diskkkkkkkkkkkkkkkkkkkkkkk
        r.save(filename=f"results{i}.jpg")
        
        # View results
        sorted_tensor = r.boxes.data[r.boxes.data[:, 0].argsort()]
        l.append(sorted_tensor[:, -2:].tolist())
        
    ans = []
    for arr in l:
        temp=""
        for i in range(len(arr)):
            temp=temp+classes[int(arr[i][1])]
        ans.append([temp])
            
    for i in ans:
        print(i)

def print_answer_with_prob(results):
    # Visualize the results
    sorted_tensor = 0
    l=[]
    for i, r in enumerate(results):
        sorted_tensor = r.boxes.data[r.boxes.data[:, 0].argsort()]
        l.append(sorted_tensor[:, -2:].tolist())
        
    for arr in l:
        for i in range(len(arr)):
            # print(classes[int(arr[i])])
            arr[i][1] = classes[int(arr[i][1])]
            
    for i in l:
        print(i)

bestmodel=YOLO('C:\\Users\\Wang Yanchi\\Desktop\\coding\\textrecgnition\\runs\\detect\\yolov8_license_plate6\\weights\\best.pt')

results = bestmodel.predict(source='C:\\Users\\Wang Yanchi\\Desktop\\coding\\textrecgnition\\test\\images', iou=0.3, conf=0.5, show_conf=False)
print_final_answer(results)
