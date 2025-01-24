import cv2

def adjust_overlay_opacity(input_video, output_video, alpha=0.8):
    """
    Adjusts the overlay opacity of an already processed video.

    Parameters:
        input_video (str): Path to the input video.
        output_video (str): Path to save the adjusted video.
        alpha (float): Blending ratio for the original frame (0.0 to 1.0).

    Returns:
        None
    """
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simulate adjusted overlay opacity (adjust `alpha` as needed)
        adjusted_frame = cv2.addWeighted(frame, alpha, frame, 0, 0)
        out.write(adjusted_frame)

    cap.release()
    out.release()

input_video = "output.mp4"  
output_video = "output_adjusted_opacity.mp4"  # Save the adjusted video here
adjust_overlay_opacity(input_video, output_video, alpha=0.8)
