from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps

input_directory = Path("/mnt/c/Users/13144/Documents/sharpened_blueberry_flower_photos/blueberry_flower_photos/dataset1")  # Input directory containing original images
output_directory = Path("testing_all_120")  # Directory to save resized images

def process_image(image_file):
    try:
        # Open the image
        img = Image.open(image_file)
        
        # Correct orientation using EXIF metadata
        img = ImageOps.exif_transpose(img)
        
        # Get original dimensions
        width, height = img.size
        
        # Calculate new dimensions (2 times smaller)
        new_width = width // 2
        new_height = height // 2
        
        # Resize the image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Construct the output file path
        output_file = output_directory / f"{image_file.stem}_smaller{image_file.suffix}"
        
        # Save the resized image
        img_resized.save(output_file)  
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

# Collect all image files in the input directory
image_files = [image_file for image_file in input_directory.rglob("*.*") if image_file.is_file()]

# Use ProcessPoolExecutor to parallelize
with ProcessPoolExecutor() as executor:
    # Map the process_image function to all image files
    executor.map(process_image, image_files)
print("All images resized and saved.")
