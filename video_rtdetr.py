import torch
import cv2
from PIL import Image, ImageDraw
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import numpy as np
import time

# Initialize the image processor and model
# Load the image processor and model
pretrained = 'PekingU/rtdetr_r18vd'
# pretrained = 'PekingU/rtdetr_r50vd'
# pretrained = 'PekingU/rtdetr_r101vd'
# pretrained = 'PekingU/rtdetr_r101vd_coco_o365'

image_processor = RTDetrImageProcessor.from_pretrained(pretrained)
model = RTDetrForObjectDetection.from_pretrained(pretrained)

# Open the video file
video_path = 'park.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

start_time = time.time()

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    inputs = image_processor(images=image, return_tensors="pt")

    # Perform object detection
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the outputs
    results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)
    print(f'{len(results[0]["labels"])} detected')

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score, label = score.item(), label_id.item()
            box = [round(i, 2) for i in box.tolist()]
            print(f"{model.config.id2label[label]}: {score:.2f} {box}")
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{model.config.id2label[label]}: {score:.2f}", fill="red")

    # Convert the image back to a format suitable for displaying
    frame_with_boxes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Display the frame with bounding boxes
    cv2.imshow('Object Detection', frame_with_boxes)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")