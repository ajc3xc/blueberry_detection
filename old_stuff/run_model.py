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
        most_common_plant_types = result[0]['plant_types']  # Labels per detected flower
        
        # Flatten the list of plant types and count occurrences for each stage
        flattened = [stage for sublist in most_common_plant_types for stage in sublist]
        counts = Counter(flattened)
        
        # Get counts for each stage explicitly
        stage_counts = {f"Stage{i}": counts.get(f"Stage{i}", 0) for i in range(1, 6)}
        
        # Total flowers detected
        total_flowers_detected = sum(stage_counts.values())
        
        print(f"Processed: {image_file}")
        return {
            "File Path": str(image_file.name),
            **stage_counts,  # Add stage counts to the result
            "Total Flowers Detected": total_flowers_detected
        }
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        return {
            "File Path": str(image_file.name),
            **{f"Stage{i}": 0 for i in range(1, 6)},  # Default to 0 for all stages
            "Total Flowers Detected": 0
        }

if __name__ == "__main__":
    # Main execution
    folder_path = "testing_all_120"

    results = []  # To store results for the DataFrame
    folder = Path(folder_path)  # Convert to pathlib Path object
    image_files = [image for image in folder.rglob('*.*') if image.is_file()]  # List all image files

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the thread pool
        futures = {executor.submit(process_image, image): image for image in image_files}
        
        # Collect results as they are completed
        for future in as_completed(futures):
            results.append(future.result())

    # Create and save the DataFrame
    df = pd.DataFrame(results)
    output_path = "processed_results_all_120.csv"  # Output file name
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
