import cv2
import torch
import numpy as np
from deep_sort import DeepSort

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Initialize DeepSORT
deep_sort = DeepSort("path_to_deepsort_checkpoint.ckpt")

# Set the model to evaluation mode and use it for inference
model.eval()

# Open the video file
cap = cv2.VideoCapture('people.mp4')
if not cap.isOpened():
    print("Error opening video file")

# Get video dimensions and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourc(*'mp4v'), fps, (width, height))

# Initialize an overlay for drawing the trails
trail_overlay = np.zeros((height, width, 3), dtype=np.uint8)

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform detection
    results = model(frame)
    
    # Extract only person detections
    detections = results.xyxy[0]  # Detections in xyxy format
    # Filter for persons only and extract required information for DeepSORT
    detections = detections[detections[:, 5] == 0]  # Assuming '0' is the class index for persons
    boxes = detections[:, :4]  # Bounding box coordinates
    scores = detections[:, 4]  # Confidence scores

    # Convert detections to the format expected by DeepSORT
    # DeepSORT requires [x1, y1, x2, y2, confidence]
    tracker_input = np.column_stack((boxes, scores))
    
    # Update DeepSORT with current frame and detections
    outputs = deep_x.sort.update(tracker_input, frame)
    
    # Draw tracking results
    for track in outputs:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Draw a dot at the centroid of each tracked pedestrian
        cv2.circle(trail_overlay, centroid, 5, (0, 0, 255), -1)  # Red dot at the centroid

    # Add the overlay with the trails to the current frame
    frame = cv2.addWeighted(frame, 1, trail_overlay, 0.5, 0)

    # Write the frame with the detections and trails
    out.write(frame)

    # Display the resulting frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
