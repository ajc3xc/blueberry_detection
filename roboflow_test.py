from inference_sdk import InferenceHTTPClient
from collections import Counter
import pandas as pd
from pathlib import Path

# Your Roboflow API key
api_key = ""

# Initialize the inference client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

# Define the recursive processing function
def process_images_in_folder(folder_path):
    results = []  # To store results for the DataFrame
    folder = Path(folder_path)  # Convert to pathlib Path object
    
    # Iterate recursively through all image files
    for image_file in folder.rglob('*.*'):  # Adjust glob pattern as needed for image types
        if image_file.is_file():  # Ensure it's a file
            try:
                # Run the workflow
                result = client.run_workflow(
                    workspace_name="blueberry-stage-detection",
                    workflow_id="blueberry-stage-detection-workflow",
                    images={"image": str(image_file)}
                )
                
                # Extract results
                print(len(result))
                most_common_plant_type = result[0]['plant_types']
                total_flowers_detected = sum(result[0]['flowers_detected'])
                
                # Flatten the list of plant types and count occurrences
                flattened = [stage for sublist in most_common_plant_type for stage in sublist]
                counts = Counter(flattened)
                print(len(counts.most_common(1)))
                most_common_stage = counts.most_common(1)[0][0]  # Find the most common stage
                
                # Append to results
                results.append({
                    "File Path": str(image_file.name),
                    "Most Common Stage": most_common_stage,
                    "Total Flowers Detected": total_flowers_detected
                })
                
                print(f"Processed: {image_file}")
            
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results.append({
                    "File Path": str(image_file.name),
                    "Most Common Stage": "Error",
                    "Total Flowers Detected": 0
                })
    
    return results

folder_path = "testing"
results = process_images_in_folder(folder_path)

# Create and save the DataFrame
df = pd.DataFrame(results)
output_path = "processed_results_all_120.csv"  # Output file name
df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
