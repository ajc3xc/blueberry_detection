from inference_sdk import InferenceHTTPClient
from collections import Counter

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="lSKkd4HovStNs61pYXBH"
)

result = client.run_workflow(
    workspace_name="blueberry-stage-detection",
    workflow_id="blueberry-stage-detection-workflow",
    images={
        "image": "testing/sharpened_smaller.jpg"
    }
)

most_common_plant_type = result[0]['plant_types']
total_flowers_detected = sum(result[0]['flowers_detected'])

# Step 1: Flatten the list
flattened = [stage for sublist in most_common_plant_type for stage in sublist]


# Step 2: Count occurrences of each stage
counts = Counter(flattened)

# Step 3: Find the most common stage
most_common_stage = counts.most_common(1)[0][0]

print("Most common stage:", most_common_stage)
print("Total flowers detected:", total_flowers_detected)