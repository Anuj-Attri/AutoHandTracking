import cv2
import mediapipe as mp
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

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
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        raise ValueError("Failed to read the video or the video is empty.")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    bounding_boxes = []

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape

        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_coords) * w)
            y_min = int(min(y_coords) * h)
            x_max = int(max(x_coords) * w)
            y_max = int(max(y_coords) * h)

            bounding_boxes.append([x_min, y_min, x_max, y_max])

    cap.release()

    return bounding_boxes, frame

def track_hands(video_path, bounding_boxes, output_path):
    """
    Track hands in a video using SAM 2 based on initial bounding boxes.

    Parameters:
        video_path (str): Path to the input video.
        bounding_boxes (list): List of bounding boxes from the first frame.
        output_path (str): Path to save the output video with masks.
    """
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path)

        # Add prompts using bounding boxes
        obj_ids = list(range(1, len(bounding_boxes) + 1))  # Unique object IDs
        frame_idx = 0  # Prompts are based on the first frame
        for obj_id, box in zip(obj_ids, bounding_boxes):
            predictor.add_new_points_or_box(state, box=box, obj_id=obj_id, frame_idx=frame_idx)

        # Process video frames using the generator
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            ret, frame = cap.read()
            if not ret:
                break

            # Overlay the masks on the frame
            for mask in masks:
                mask = mask.cpu().numpy().astype(np.uint8).squeeze() * 255  # Ensure 2D
                colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

            out.write(frame)

    cap.release()
    out.release()




if __name__ == "__main__":
    video_path = "test.mp4"
    output_path = "output.mp4"

    # Step 1: Detect hands in the first frame
    boxes, first_frame = detect_hands_in_first_frame(video_path)

    # Step 2: Track hands using SAM 2
    track_hands(video_path, boxes, output_path)
