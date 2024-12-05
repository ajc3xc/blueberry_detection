from inference_sdk import InferenceHTTPClient
from collections import Counter
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from key import key
import sys

# Initialize the inference client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=key
)

# Function to process a single image and get predictions
def process_image(image_file):
    # Run the workflow
    result = client.run_workflow(
        workspace_name="blueberry-stage-detection",
        workflow_id="blueberry-stage-detection-workflow",
        images={"image": str(image_file)}
    )
    
    # Extract results
    most_common_plant_types = result[0]['plant_types']  # Labels per detected flower
    
    #print(most_common_plant_types)
    
    # Flatten the list of plant types and count occurrences for each stage
    flattened = [stage for sublist in most_common_plant_types for stage in sublist]
    counts = Counter(flattened)
    
    most_common_stage = counts.most_common(1)[0][0]  # Find the most common stage
    
    # Get counts for each stage explicitly
    stage_counts = {f"Pred_Stage{i}": counts.get(f"Stage{i}", 0) for i in range(1, 6)}
    
    # Total flowers detected
    total_flowers_detected = sum(stage_counts.values())
    
    #print(f"Processed: {image_file}")
    return {
        "File Path": image_file,
        **stage_counts,  # Add stage counts to the result
        "Pred_Most_Common_Stage": most_common_stage,
        "Pred_Total_Flowers": total_flowers_detected
    }

# Main execution
if __name__ == "__main__":
    # Directories and files
    input_directory = Path("test_labels")  # Replace with your directory path
    folder_path = Path("testing_all_120")  # Directory with test images
    output_csv = "ground_truth_results.csv"  # Output CSV file

    # Collect ground truth data
    ground_truth_data = []
    for file in input_directory.glob("P*.txt"):
        p_number = file.stem.split("_")[0]  # Extract P{number}
        # Read the file and parse the stage numbers
        with file.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0 and parts[0].isdigit():
                    stage = int(parts[0])
                    if 0 <= stage <= 4:
                        ground_truth_data.append({
                            "File Path": str(folder_path / f"{p_number}_smaller.JPG"),
                            "Ground Truth Stage": stage + 1  # Stage numbers are 1-indexed
                        })
    ground_truth_df = pd.DataFrame(ground_truth_data)
    
    # Count occurrences of each stage for each P_Number
    stage_counts_df = ground_truth_df.groupby(["File Path", "Ground Truth Stage"]).size().unstack(fill_value=0)

    # Rename the columns for clarity
    stage_counts_df.columns = [f"Ground_Stage{i}" for i in stage_counts_df.columns]
    stage_counts_df["Ground_Combined"] = stage_counts_df.sum(axis=1)
    relevant_columns = [col for col in stage_counts_df.columns if col != "Ground_Combined"]
    # Calculate the most common stage for each row
    stage_counts_df["Ground_Most_Common_Stage"] = stage_counts_df[relevant_columns].idxmax(axis=1).str.replace("Ground_", "")
    stage_counts_df.reset_index(inplace=True)
    image_files = stage_counts_df["File Path"].tolist()

    # Process predictions
    predictions = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        predictions = list(executor.map(process_image, image_files))

    predictions_df = pd.DataFrame(predictions)

    # Combine ground truth and predictions into a single DataFrame
    combined_df = pd.merge(
        stage_counts_df,
        predictions_df,
        on="File Path",
        how="left"
    )
    
    combined_df = combined_df[['File Path', "Ground_Stage1", "Pred_Stage1", "Ground_Stage2", "Pred_Stage2", "Ground_Stage3", "Pred_Stage3", "Ground_Stage4", "Pred_Stage4", "Ground_Stage5", "Pred_Stage5", "Ground_Combined", "Pred_Total_Flowers", "Ground_Most_Common_Stage", "Pred_Most_Common_Stage"]]

    # Save combined results to CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined results saved to {output_csv}")
