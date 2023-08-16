import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms

# Load and preprocess image with data augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])


# Load and preprocess image
image = cv2.imread('../images/cropped/1_1.jpg')
image = transform(image)
image = image.unsqueeze(0)
#image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float().unsqueeze(0)

# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(weights=True)
model.eval()

# Perform inference
with torch.no_grad():
    predictions = model(image)[0]

# Extract masks, bounding boxes, and class IDs
masks = predictions['masks'].cpu().numpy()
boxes = predictions['boxes'].cpu().numpy()
class_ids = predictions['labels'].cpu().numpy()

# Calculate leaf count
leaf_count = len(boxes)

# Calculate leaf area
leaf_area = np.sum(np.sum(masks, axis=0))

print(f"Leaf Count: {leaf_count}")
print(f"Leaf Area: {leaf_area}")