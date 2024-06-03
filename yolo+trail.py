import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

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
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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
    results = results.pandas().xyxy[0]  # DataFrame with bounding boxes, confidences, etc.
    results = results[results['name'] == 'person']  # Filter for persons only

    # Update paths and draw bounding boxes
    for index, row in results.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {int(row["confidence"]*100)}%', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        # Draw a dot at the centroid of each detected pedestrian on the overlay
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
