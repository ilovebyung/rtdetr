from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolov8n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("yolov8n.onnx")

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")


from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# model.export(format="onnx")
onnx_model = RTDETR("rtdetr-l.onnx")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=10, imgsz=320)
results = onnx_model.train(data="coco8.yaml", epochs=10, imgsz=320)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("byung.jpg")
results = model("https://ultralytics.com/images/bus.jpg")
results = onnx_model("https://ultralytics.com/images/bus.jpg")

#######################
model = RTDETR("yolov8m-rtdetr.yaml")
model.train(data='coco128.yaml', batch=2)
model.save("yolov8m-rtdetr.pt")