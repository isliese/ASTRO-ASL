import cv2 # type: ignore
import mediapipe as mp # type: ignore

# Load model
# model = keras.models.load_model('models/model.keras')

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    # Show the original webcam feed
    cv2.imshow("Webcam Feed", frame)

    # Check if hands are detected
    if result.multi_hand_landmarks:

        for landmarks in result.multi_hand_landmarks:
            # Get the bounding box around the hand
            x_min = min([lm.x for lm in landmarks.landmark]) * frame.shape[1]
            x_max = max([lm.x for lm in landmarks.landmark]) * frame.shape[1]
            y_min = min([lm.y for lm in landmarks.landmark]) * frame.shape[0]
            y_max = max([lm.y for lm in landmarks.landmark]) * frame.shape[0]

            # Calculate the center of the hand
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)

            # Define the zoomed-in region
            size = 224
            x1 = max(center_x - size // 2, 0)
            y1 = max(center_y - size // 2, 0)
            x2 = min(center_x + size // 2, frame.shape[1])
            y2 = min(center_y + size // 2, frame.shape[0])

            # Extract the region
            zoomed_in_hand = frame[y1:y2, x1:x2]

            # Display the zoomed-in region
            cv2.imshow("Zoomed Hand", zoomed_in_hand)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
