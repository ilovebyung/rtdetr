# medium model
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics import RTDETR

# from PIL
image = cv2.imread("1-park.jpg")



model = YOLO("models/yolov8m.pt")
results = model.predict(source=image, save=True)  # save plotted images
 


# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")
results = model.predict(source=image, save=True) 

 