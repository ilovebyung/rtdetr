from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # 'n' for nano model, you can use 's', 'm', 'l', or 'x' for other sizes

# Read the image
image = cv2.imread('park.jpg')

# Perform object detection
results = model(image)

for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        # Get the coordinates
        x1, y1, x2, y2 = box.xyxy[0].astype(int)

        # Get the class and confidence
        class_id = box.cls[0].astype(int)
        conf = box.conf[0]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put class name and confidence
        label = f'{model.names[class_id]} {conf:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow('YOLOv8 Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Optionally, save the output image
# cv2.imwrite('park_detected.jpg', image)