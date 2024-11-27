from inference_sdk import InferenceHTTPClient
from collections import Counter
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Your Roboflow API key
api_key = ""

# Initialize the inference client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

# Function to process a single image
def process_image(image_file):
    try:
        # Run the workflow
        result = client.run_workflow(
            workspace_name="blueberry-stage-detection",
            workflow_id="blueberry-stage-detection-workflow",
            images={"image": str(image_file)}
        )
        
        # Extract results
        most_common_plant_type = result[0]['plant_types']
        total_flowers_detected = sum(result[0]['flowers_detected'])
        
        # Flatten the list of plant types and count occurrences
        flattened = [stage for sublist in most_common_plant_type for stage in sublist]
        counts = Counter(flattened)
        most_common_stage = counts.most_common(1)[0][0]  # Find the most common stage
        
        print(f"Processed: {image_file}")
        return {
            "File Path": str(image_file.name),
            "Most Common Stage": most_common_stage,
            "Total Flowers Detected": total_flowers_detected
        }
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        return {
            "File Path": str(image_file.name),
            "Most Common Stage": "None",
            "Total Flowers Detected": 0
        }

# Function to process images in a folder with parallelization
def process_images_in_folder_parallel(folder_path, max_workers=4):
    results = []  # To store results for the DataFrame
    folder = Path(folder_path)  # Convert to pathlib Path object
    image_files = [image for image in folder.rglob('*.*') if image.is_file()]  # List all image files
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the thread pool
        futures = {executor.submit(process_image, image): image for image in image_files}
        
        # Collect results as they are completed
        for future in as_completed(futures):
            results.append(future.result())
    
    return results

# Main execution
folder_path = "testing_all_120"
results = process_images_in_folder_parallel(folder_path, max_workers=8)  # Adjust max_workers based on your CPU/GPU capabilities

# Create and save the DataFrame
df = pd.DataFrame(results)
output_path = "processed_results_all_120.csv"  # Output file name
df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")