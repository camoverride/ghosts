import os
import subprocess
import time
import yaml
import cv2
import numpy as np

# Read data from the config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
WIDTH = config["WIDTH"]
HEIGHT = config["HEIGHT"]
PLAY_DIR = config["PLAY_DIR"]
CAPTURE_DURATION = config["CAPTURE_DURATION"]
FPS = config["FPS"]

# Set the DISPLAY environment variable for the current process
os.environ["DISPLAY"] = ":0"

# Set to fullscreen
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def get_latest_video_path(play_dir):
    """Returns the path of the most recent video in the directory."""
    file_paths = list(reversed(sorted([os.path.join(play_dir, f) for f in os.listdir(play_dir)])))
    return file_paths[0] if len(file_paths) > 0 else None

def is_video_file_ready(video_path):
    """Check if the video file is ready to be played."""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, _ = cap.read()
        cap.release()
        return ret  # Returns True if the first frame is successfully read
    except:
        return False

def play_video(video_path):
    """Play the video in a loop until a new video is available."""
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
            continue

        cv2.imshow("window", frame)
        key = cv2.waitKey(int(1000 / FPS))  # Adjust this to match the desired FPS

        # Check if a new video is available every few frames
        if frame_counter % FPS == 0:  # Check every second
            new_video_path = get_latest_video_path(PLAY_DIR)
            if new_video_path != video_path and is_video_file_ready(new_video_path):
                print(f"New video detected: {new_video_path}")
                cap.release()
                return new_video_path

        frame_counter += 1

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)  # Exit the entire program

def main():
    last_video_path = None

    # Configure the screen properly (run only once)
    subprocess.run("WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90",
                   shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        latest_video_path = get_latest_video_path(PLAY_DIR)

        # Check if a new video has been created and if it's ready to play
        if latest_video_path != last_video_path and is_video_file_ready(latest_video_path):
            last_video_path = latest_video_path
            print(f"Loading new video: {last_video_path}")
        
        # Play the current video, checking for a new one
        last_video_path = play_video(last_video_path)

if __name__ == "__main__":
    main()
