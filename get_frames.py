
import cv2
import os

def extract_frames(video_path, output_folder):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get video information
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each frame and save as JPEG
    for frame_number in range(total_frames):
        # Read the frame
        ret, frame = video_capture.read()

        # If the frame is not read properly, break the loop
        if not ret:
            break

        # Save the frame as JPEG
        frame_name = f"{output_folder}/frame_{frame_number + 1:04d}.jpg"
        cv2.imwrite(frame_name, frame)

    # Release the video capture object
    video_capture.release()

    print(f"Frames extracted and saved in {output_folder}")

if __name__ == "__main__":
    files = os.listdir(os.path.join(os.path.abspath(os.path.dirname(__file__)),"Avenue_Dataset/Avenue Dataset/testing_videos"))
    print(files)
    for f in files:
        # Specify the path to the video file

        video_path = f"./Avenue_Dataset/Avenue Dataset/testing_videos/{f}"

        # Extract video name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Specify the output folder
        output_folder = f"avenue/testing/frames/{video_name}_frames"

        # Call the function to extract frames
        extract_frames(video_path, output_folder)