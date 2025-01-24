import cv2
import mediapipe as mp
import numpy as np

def detect_hands_in_first_frame(video_path):
    """
    Detects hands in the first frame of a video using MediaPipe.

    Parameters:
        video_path (str): Path to the input video.

    Returns:
        list: A list of bounding boxes for detected hands in the first frame.
              Each bounding box is represented as [x_min, y_min, x_max, y_max].
        numpy.ndarray: The first frame of the video.
    """
    # Initialize MediaPipe HandLandmarker
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    # Capture video and read the first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        raise ValueError("Failed to read the video or the video is empty.")

    # Convert the frame to RGB (MediaPipe expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)

    bounding_boxes = []

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape

        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks and calculate bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_coords) * w)
            y_min = int(min(y_coords) * h)
            x_max = int(max(x_coords) * w)
            y_max = int(max(y_coords) * h)

            bounding_boxes.append([x_min, y_min, x_max, y_max])

    # Release the video capture object
    cap.release()

    return bounding_boxes, frame

if __name__ == "__main__":
    video_path = "test.mp4"  # Replace with the path to your video
    boxes, first_frame = detect_hands_in_first_frame(video_path)

    print("Detected bounding boxes:", boxes)

    # Convert bounding boxes into SAM 2 prompts (e.g., center points)
    sam2_prompts = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        sam2_prompts.append([center_x, center_y])

    print("SAM 2 Prompts (Center Points):", sam2_prompts)

    # Visualize the bounding boxes on the first frame for verification
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(first_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("First Frame with Bounding Boxes", first_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
