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
    
    # Get counts for each stage explicitly
    stage_counts = {f"Pred_Stage{i}": counts.get(f"Stage{i}", 0) for i in range(1, 6)}
    
    # Total flowers detected
    total_flowers_detected = sum(stage_counts.values())
    
    return {
        "File Path": image_file,
        **stage_counts,  # Add stage counts to the result
        "Pred_Total_Flowers": total_flowers_detected
    }

# Main execution
if __name__ == "__main__":
    # Directories and files
    input_directory = Path("test_labels")
    folder_path = Path("testing_all_120")
    output_csv = "ground_truth_results_with_confusion.csv"

    # Collect ground truth data
    ground_truth_data = []
    for file in input_directory.glob("P*.txt"):
        p_number = file.stem.split("_")[0]
        with file.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0 and parts[0].isdigit():
                    stage = int(parts[0])
                    if 0 <= stage <= 4:
                        ground_truth_data.append({
                            "File Path": folder_path / f"{p_number}_smaller.JPG",
                            "Ground Truth Stage": stage + 1
                        })
    ground_truth_df = pd.DataFrame(ground_truth_data)

    # Count occurrences of each stage for each P_Number
    stage_counts_df = ground_truth_df.groupby(["File Path", "Ground Truth Stage"]).size().unstack(fill_value=0)
    stage_counts_df.columns = [f"Ground_Stage{i}" for i in stage_counts_df.columns]
    stage_counts_df["Ground_Combined"] = stage_counts_df.sum(axis=1)
    relevant_columns = [col for col in stage_counts_df.columns if col != "Ground_Combined"]
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

    # Calculate confusion matrix for each image
    confusion_data = []
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

        confusion_data.append({"File Path": row["File Path"], **confusion_matrix})

    confusion_df = pd.DataFrame(confusion_data)

    # Merge confusion data back into the combined DataFrame
    combined_df = pd.merge(combined_df, confusion_df, on="File Path", how="left")

    # Save the combined DataFrame with confusion matrices to a CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Ground truth results with confusion matrices saved to {output_csv}")
