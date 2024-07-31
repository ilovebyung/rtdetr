import torch
from PIL import Image, ImageDraw
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import matplotlib.pyplot as plt

# pretrained = 'PekingU/rtdetr_r18vd'
pretrained = 'PekingU/rtdetr_r50vd'
# pretrained = 'PekingU/rtdetr_r101vd'
image_processor = RTDetrImageProcessor.from_pretrained(pretrained)
model = RTDetrForObjectDetection.from_pretrained(pretrained)

# Load the image
image = Image.open('park.jpg')

# Preprocess the image
inputs = image_processor(images=image, return_tensors="pt")

# Perform object detection
with torch.no_grad():
    outputs = model(**inputs)

# Post-process the outputs
results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)
len(results[0]['labels'])

# Draw bounding boxes on the image
draw = ImageDraw.Draw(image)
for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{model.config.id2label[label]}: {score:.2f}", fill="red")

# Display the image with bounding boxes
plt.figure(figsize=(12, 8))
plt.imshow(image)
plt.axis('off')
plt.show()