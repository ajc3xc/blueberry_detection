from PIL import Image, ImageDraw
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Directories for images and bounding box text files
image_dir = Path("testing_all_120")
bbox_dir = Path("test_labels")
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

# Custom colors for each stage
stage_colors = {
    1: "#FF0000",  # Red for Stage 1
    2: "#00FF00",  # Green for Stage 2
    3: "#0000FF",  # Blue for Stage 3
    4: "#FFFF00",  # Yellow for Stage 4
    5: "#FF00FF"   # Magenta for Stage 5
}

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image_file):
    base_name = image_file.stem
    bbox_files = list(bbox_dir.glob(f"{base_name}*.txt"))

    if not bbox_files:
        print(f"No bounding box files found for {image_file}")
        return

    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)

    for bbox_file in bbox_files:
        with bbox_file.open("r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:  # Format: x_min y_min x_max y_max stage
                    x_min, y_min, x_max, y_max, stage = map(float, parts)
                    stage = int(stage)
                    color = stage_colors.get(stage, "#FFFFFF")  # Default to white
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

    output_path = output_dir / image_file.name
    image.save(output_path)

# Process all images using ProcessPoolExecutor
image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

with ProcessPoolExecutor() as executor:
    executor.map(draw_bounding_boxes, image_files)

print("Bounding boxes have been drawn and saved to the output directory.")
