# Import necessary libraries
import os
import requests
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import supervision as sv
import numpy as np

# Function to download model weights
def download_file(url, dest):
    if not os.path.isfile(dest):
        print(f"Downloading {url} to {dest}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print("Download complete.")
        else:
            raise Exception(f"Failed to download file: {response.status_code}")
    else:
        print(f"File already exists: {dest}")

# Download SAM model weights if not already downloaded
weights_dir = os.path.expanduser("~/weights")
os.makedirs(weights_dir, exist_ok=True)
weights_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
weights_path = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
download_file(weights_url, weights_path)

# Load SAM Model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=weights_path).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)

# Example image path
IMAGE_PATH = r"G:\Sanjayram R\postgen\images\pepsi.jfif"

# Variables to store the bounding box coordinates
drawing = False
ix, iy = -1, -1
boxes = []

# Mouse callback function to draw rectangles
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        boxes.append((ix, iy, x - ix, y - iy))
        cv2.imshow("Image", img)

# Load image and set mouse callback
img = cv2.imread(IMAGE_PATH)
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_rectangle)

# Display image and wait for user to draw bounding boxes
print("Draw bounding boxes with the mouse. Press 'q' to quit.")
while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Print the coordinates of the drawn bounding boxes
print("Bounding boxes:", boxes)

def draw_bounding_boxes(image_path, boxes):
    image = cv2.imread(image_path)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display image briefly (100 milliseconds) and continue
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(100)
    cv2.destroyAllWindows()


draw_bounding_boxes(IMAGE_PATH, boxes)
# Assuming you want to convert the first box in the list
# default_box is going to be used if you will not draw any box on image above
default_box = {'x': 68, 'y': 247, 'width': 555, 'height': 678, 'label': ''}
box = boxes[0] if boxes else default_box
# Convert box coordinates to numpy array [x1, y1, x2, y2]
box_np = np.array([
    box[0],  # x1
    box[1],  # y1
    box[0] + box[2],  # x2 = x1 + width
    box[1] + box[3]   # y2 = y1 + height
])

print("Converted box:", box_np)

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

mask_predictor.set_image(image_rgb)

masks, scores, logits = mask_predictor.predict(
    box=box_np,
    multimask_output=True
)

box_annotator = sv.BoxAnnotator(color=sv.Color.red())
mask_annotator = sv.MaskAnnotator(color=sv.Color.red(), color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=masks),
    mask=masks
)
detections = detections[detections.area == np.max(detections.area)]

source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[source_image, segmented_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)
# Display the images
sv.plot_images_grid(
    images=[source_image, segmented_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)

# Save the segmented image
segmented_image_pil = Image.fromarray(segmented_image)
segmented_image_pil.save('segmented_output.jpg')

# Save individual masks
for i, mask in enumerate(masks):
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))  # Convert mask to binary image
    mask_pil.save(f'mask_{i}.png')

# Display masks
sv.plot_images_grid(
    images=masks,
    grid_size=(1, len(masks)),
    size=(16, 4)
)
segmented_images = []
for i, mask in enumerate(masks):
    # Create an empty image with the same dimensions as the original
    empty_image = np.zeros_like(image_rgb)
    # Copy the segmented part from the original image
    segmented_part = np.where(mask[..., None], image_rgb, empty_image)
    segmented_images.append(segmented_part)

# Save segmented images
from PIL import Image

for i, segmented_image in enumerate(segmented_images):
    segmented_image_pil = Image.fromarray(segmented_image)
    segmented_image_pil.save(f'segmented_output_{i}.png')

# Display masks
sv.plot_images_grid(
    images=[Image.fromarray(img) for img in segmented_images],
    grid_size=(1, len(segmented_images)),
    size=(16, 4)
)
print('DONE')
