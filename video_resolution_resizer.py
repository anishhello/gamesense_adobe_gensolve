import cv2

def resize_video(input_path, output_path, target_width=1280, target_height=720):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    
    # Create a VideoWriter object to write the resized video
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    while True:
        ret, frame = cap.read()  # Read frame by frame
        if not ret:
            break
        
        # Resize the frame to the target resolution
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        # Write the resized frame to the output video
        out.write(resized_frame)
    
    # Release resources
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

# # Example usage
# input_video_path = '/content/Rafael Nadal vs Novak Djokovic - Final Highlights I Roland-Garros 2020 (1).mp4'
# output_video_path = 'resized_video.mp4'
# resize_video(input_video_path, output_video_path)