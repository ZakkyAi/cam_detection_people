import cv2

# Paths to the model files (update with the correct path if needed)
prototxt_path = r"D:\python\cam_detection\models\MobileNet-SSD\deploy.prototxt"
model_path = r"D:\python\cam_detection\models\MobileNet-SSD\mobilenet_iter_73000.caffemodel"

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Define the classes that MobileNet SSD can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Create a named window to enable full screen
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the frame's height and width
    (h, w) = frame.shape[:2]
    
    # Prepare the frame for the network (resize to 300x300 for detection)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Perform forward pass through the network
    net.setInput(blob)
    detections = net.forward()

    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only process if confidence > threshold
        if confidence > 0.4:
            # Get the class index (1 is "person")
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            # If the detected class is "person"
            if label == "person":
                # Get the bounding box for the detected person
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # Draw a bounding box around the detected person
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label_text = f"{label}: {confidence * 100:.2f}%"
                cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the original aspect ratio
    cv2.imshow("Frame", frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the display window
cap.release()
cv2.destroyAllWindows()
