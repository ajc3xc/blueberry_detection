from PIL import Image, ImageDraw
from pathlib import Path

# Directories for images and bounding box text files
image_dir = Path("test/images")
bbox_dir = Path("test/labels")
output_dir = Path("ground_truth_inferences")
output_dir.mkdir(exist_ok=True)
clipped_output_dir = Path("ground_truth_bboxes_clipped")
clipped_output_dir.mkdir(exist_ok=True)
unclipped_output_dir = Path("ground_truth_inferences_unclipped")
unclipped_output_dir.mkdir(exist_ok=True)

# Function to convert YOLOv1 normalized bounding box format to pixel coordinates
def yolo_to_pixel_coords(bbox, image_width, image_height):
    x_center, y_center, width, height = bbox
    x_min = (x_center - width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    x_max = (x_center + width / 2) * image_width
    y_max = (y_center + height / 2) * image_height
    return x_min, y_min, x_max, y_max

# Function to process a single image
def process_image(image_path):
    base_name = image_path.stem
    bbox_file = bbox_dir / f"{base_name}.txt"

    # Skip if no bounding box file exists
    if not bbox_file.exists():
        print(f"No bounding box file for {base_name}")
        return

    # Open the image
    image = Image.open(str(image_path))
    image_width, image_height = image.size

    # Initialize stage0 ROI and label count dictionary
    stage0_roi = None
    label_counts = {stage: 0 for stage in range(6)}  # Assuming stages 0-5
    clipped_annotations = []
    unclipped_annotations = []

    # Read bounding box data
    with bbox_file.open("r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:  # YOLOv1 format: class x_center y_center width height
                stage, x_center, y_center, width, height = map(float, parts)
                stage = int(stage)
                x_min, y_min, x_max, y_max = yolo_to_pixel_coords([x_center, y_center, width, height], image_width, image_height)

                if stage == 0:
                    stage0_roi = (x_min, y_min, x_max, y_max)
                else:
                    label_counts[stage] += 1

                # Check if the annotation is within stage0 ROI
                if stage0_roi:
                    roi_x_min, roi_y_min, roi_x_max, roi_y_max = stage0_roi
                    unclipped_annotations.append(line.strip())
                    if (roi_x_min <= x_min <= roi_x_max and
                        roi_x_min <= x_max <= roi_x_max and
                        roi_y_min <= y_min <= roi_y_max and
                        roi_y_min <= y_max <= roi_y_max):
                        clipped_annotations.append(line.strip())

    # Clip the image to stage0 ROI if it exists
    if stage0_roi:
        x_min, y_min, x_max, y_max = map(int, stage0_roi)
        clipped_image = image.crop((x_min, y_min, x_max, y_max))

        # Save the clipped image
        clipped_output_path = clipped_output_dir / f"{base_name}_clipped_stage0.jpg"
        clipped_image.save(clipped_output_path)

        # Visualize the clipped image
        clipped_image.show()

        # Export clipped annotations to a new file
        if clipped_annotations:
            clipped_annotations_path = clipped_output_dir / f"{base_name}_clipped_annotations.txt"
            with clipped_annotations_path.open("w") as anno_file:
                anno_file.write("\n".join(clipped_annotations))
    else:
        print(f"No stage0 bounding box for {base_name}")
        return

    # Save and visualize the original image with bounding boxes
    draw = ImageDraw.Draw(image)
    with bbox_file.open("r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                stage, x_center, y_center, width, height = map(float, parts)
                stage = int(stage)
                x_min, y_min, x_max, y_max = yolo_to_pixel_coords([x_center, y_center, width, height], image_width, image_height)
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

    unclipped_output_path = unclipped_output_dir / f"{base_name}_unclipped.jpg"
    image.save(unclipped_output_path)

    # Visualize the original image
    image.show()

    # Export unclipped annotations to a new file
    if unclipped_annotations:
        unclipped_annotations_path = unclipped_output_dir / f"{base_name}_unclipped_annotations.txt"
        with unclipped_annotations_path.open("w") as anno_file:
            anno_file.write("\n".join(unclipped_annotations))

    print(f"{base_name} label counts outside stage0 ROI: {label_counts}")

# Process all images
image_files = list(image_dir.glob("*.jpg"))
print(f"Processing {len(image_files)} images...")

for image_file in image_files:
    process_image(image_file)

print("Processing complete.")
