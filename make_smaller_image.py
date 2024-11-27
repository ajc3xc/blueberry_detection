from PIL import Image
from pathlib import Path

# Input and output directories
input_directory = "/mnt/c/Users/13144/Documents/sharpened_blueberry_flower_photos/blueberry_flower_photos/dataset1"  # Input directory containing original images
output_directory = "testing"  # Directory to save resized images

# Create the output directory if it doesn't exist
output_dir = Path(output_directory)
output_dir.mkdir(parents=True, exist_ok=True)

# Iterate through all image files in the input directory
input_dir = Path(input_directory)
for image_file in input_dir.rglob("*.*"):  # Adjust pattern for specific image types if needed
    if image_file.is_file():  # Ensure it's a file
        try:
            # Open the image
            img = Image.open(image_file)
            
            # Get original dimensions
            width, height = img.size
            
            # Calculate new dimensions (2 times smaller)
            new_width = width // 2
            new_height = height // 2
            
            # Resize the image
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Construct the output file path
            output_file = output_dir / f"{image_file.stem}_smaller{image_file.suffix}"
            
            # Save the resized image
            img_resized.save(output_file)
            
            print(f"Resized and saved: {output_file}")
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

