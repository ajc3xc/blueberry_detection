from PIL import Image, ImageDraw
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Directories for images and bounding box text files
image_dir = Path("test/images")
bbox_dir = Path("test/labels")
output_dir = Path("ground_truth_inferences")
output_dir.mkdir(exist_ok=True)

# Custom colors for each stage
stage_colors = {
    0: "#0000FF",  # Red for Stage 0
    1: "#C7FC00",  # Red for Stage 1
    2: "#00FFCE",  # Green for Stage 2
    3: "#8622FF",  # Blue for Stage 3
    4: "#FE0056",  # Yellow for Stage 4
    5: "#FF8000"   # Magenta for Stage 5
}

# Function to convert YOLOv1 normalized bounding box format to pixel coordinates
def yolo_to_pixel_coords(bbox, image_width, image_height):
    x_center, y_center, width, height = bbox
    x_min = (x_center - width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    x_max = (x_center + width / 2) * image_width
    y_max = (y_center + height / 2) * image_height
    return x_min, y_min, x_max, y_max

# Function to draw bounding boxes on an image
def process_image(image_path):
    base_name = image_path.stem
    p_number = image_file.stem.split("_")[0]
    bbox_files = list(bbox_dir.glob(f"{p_number}_*.txt"))
    if bbox_files: print(base_name, bbox_files, p_number)

    # Skip if no bounding box files
    if not bbox_files:
        return

    # Open the image
    image = Image.open(str(image_path))
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    # Iterate over all bounding box files for this image
    for bbox_file in bbox_files:
        with bbox_file.open("r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:  # Assuming YOLOv1 format: class x_center y_center width height
                    stage, x_center, y_center, width, height = map(float, parts)
                    stage = int(stage)
                    color = stage_colors.get(stage, "#FFFFFF")  # Default to white if stage is invalid
                    if color == "#FFFFFF":
                        print(f"Invalid stage {stage} in {bbox_file.name}")
                    x_min, y_min, x_max, y_max = yolo_to_pixel_coords([x_center, y_center, width, height], image_width, image_height)
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

    # Save the image with bounding boxes
    output_path = output_dir / f"{image_path.stem}_ground_truth.jpg"
    image.save(output_path)

# Process images in parallel
image_files = list(image_dir.glob("*.jpg"))
print(f"Processing {len(image_files)} images...")
for image_file in image_files:
    process_image(image_file)
#with ProcessPoolExecutor() as executor:
#    executor.map(process_image, image_files)

print("Bounding boxes have been drawn and saved to the output directory.")
