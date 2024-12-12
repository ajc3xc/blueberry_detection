from inference_sdk import InferenceHTTPClient
from collections import Counter
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from key import key
from PIL import Image
import numpy as np
import io
import base64
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
    
    # Convert NumPy array to a Pillow Image object
    # Decode the Base64 string to an image
    inference_image_base64 = result[0]['inferences_image'][0]
    image_data = base64.b64decode(inference_image_base64)
    image = Image.open(io.BytesIO(image_data))

    # Save the image as a JPG file
    output_path = Path("inference_images")
    output_path.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    image.save(output_path / f"{image_file.stem}_inference.jpg")
    
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
    input_directory = Path("ground_truth_labels")  # Replace with your directory path
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
                    if 1 <= stage <= 5:
                        ground_truth_data.append({
                            "File Path": folder_path / f"{p_number}_smaller.JPG",
                            "Ground Truth Stage": stage
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
    
    # Calculate confusion matrix and metrics
    confusion_data = []
    metrics_data = []
    overall_true_positives = 0
    overall_false_positives = 0
    overall_false_negatives = 0

    for _, row in combined_df.iterrows():
        confusion_matrix = {}
        for stage in range(1, 6):
            ground_truth_count = row.get(f"Ground_Stage{stage}", 0)
            prediction_count = row.get(f"Pred_Stage{stage}", 0)
            true_positive = min(ground_truth_count, prediction_count)
            false_positive = prediction_count - true_positive
            false_negative = ground_truth_count - true_positive

            confusion_matrix[f"Stage{stage}_TP"] = true_positive
            confusion_matrix[f"Stage{stage}_FP"] = false_positive
            confusion_matrix[f"Stage{stage}_FN"] = false_negative

            overall_true_positives += true_positive
            overall_false_positives += false_positive
            overall_false_negatives += false_negative

        confusion_data.append({"File Path": row["File Path"], **confusion_matrix})

    for stage in range(1, 6):
        stage_true_positive = sum(row[f"Stage{stage}_TP"] for row in confusion_data)
        stage_false_positive = sum(row[f"Stage{stage}_FP"] for row in confusion_data)
        stage_false_negative = sum(row[f"Stage{stage}_FN"] for row in confusion_data)

        precision = stage_true_positive / (stage_true_positive + stage_false_positive) if stage_true_positive + stage_false_positive > 0 else 0
        recall = stage_true_positive / (stage_true_positive + stage_false_negative) if stage_true_positive + stage_false_negative > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        metrics_data.append({
            "Stage": stage,
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1 Score": round(f1_score * 100, 2)
        })

    # Calculate overall metrics
    overall_precision = overall_true_positives / (overall_true_positives + overall_false_positives) if overall_true_positives + overall_false_positives > 0 else 0
    overall_recall = overall_true_positives / (overall_true_positives + overall_false_negatives) if overall_true_positives + overall_false_negatives > 0 else 0
    overall_f1_score = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0

    metrics_data.append({
        "Stage": "Overall",
        "Precision": round(overall_precision * 100, 2),
        "Recall": round(overall_recall * 100, 2),
        "F1 Score": round(overall_f1_score * 100, 2)
    })

    confusion_df = pd.DataFrame(confusion_data)
    metrics_df = pd.DataFrame(metrics_data)

    #metrics_df = pd.concat([metrics_df, overall_metrics], ignore_index=True)

    #metrics_df = metrics_df.round(2)
    
    #print("test")

    # Save metrics to CSV
    metrics_df.to_csv("detection_metrics_percent.csv", index=False)
    print("Detection metrics saved to detection_metrics.csv")
    
    
    # Calculate the mean for numeric columns
    mean_values = combined_df.select_dtypes(include=['number']).mean().to_frame(name='Mean').T.astype(int)
    mean_values["File Path"] = "Mean"
    combined_df = pd.concat([combined_df, mean_values], ignore_index=False)

    # Save combined results to CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Ground truth results saved to {output_csv}")

