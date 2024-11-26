#original image is too large.

from PIL import Image

# Open the image
img = Image.open("input_output_files/sharpened.JPG")

# Get original dimensions
width, height = img.size

# Calculate new dimensions (4 times smaller)
new_width = width // 2
new_height = height // 2

# Resize the image
img_resized = img.resize((new_width, new_height), Image.LANCZOS)

# Resize the image
img.save("testing/sharpened_smaller.jpg")
