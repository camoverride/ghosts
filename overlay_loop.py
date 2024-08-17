from datetime import datetime
import os
import time
import yaml

import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common


# Read data from the config.
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
WIDTH = config["WIDTH"]
HEIGHT = config["HEIGHT"]
PLAY_DIR = config["PLAY_DIR"]
CONFIDENCE_THRESHOLD = config["CONFIDENCE_THRESHOLD"]
NEW_IMAGES_MEMMAP_PATH = config["NEW_IMAGES_MEMMAP_PATH"]
ALPHA = config["ALPHA"]
CAPTURE_DURATION = config["CAPTURE_DURATION"]
FPS = config["FPS"]

# Initialize the picamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888",
                                                            "size": (WIDTH, HEIGHT)}))
picam2.start()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the TFLite model for overlaying
interpreter = make_interpreter("overlay_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# When the program starts, save two new videos to the play folder.
frame_count = int(CAPTURE_DURATION * FPS)

for _ in range(2):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{current_time}.dat"
    start_memmap_path = os.path.join(PLAY_DIR, filename)
    print(f"Creating start file {start_memmap_path}")
    memmap_shape = (frame_count, HEIGHT, WIDTH, 3)
    start_memmap = np.memmap(start_memmap_path, dtype='uint8', mode='w+', shape=memmap_shape)

    for frame_num in range(frame_count):
        frame = picam2.capture_array()
        start_memmap[frame_num] = frame

    # Finalize the memmap file
    start_memmap.flush()

# Begin the main loop
with mp_face_detection.FaceDetection(min_detection_confidence=CONFIDENCE_THRESHOLD) as face_detection:
    while True:
        # Snap two photos for temporal filtering to reduce the likelihood of false positives
        frame_1 = picam2.capture_array()
        time.sleep(0.5)
        frame_2 = picam2.capture_array()

        # Process the frames and detect faces
        results_1 = face_detection.process(frame_1)
        results_2 = face_detection.process(frame_2)

        # Get the time for filenaming
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        if 1==1:#results_1.detections and results_2.detections:
            t = datetime.now().strftime("%H-%M-%S")
            print(f"{t} - Face detected! Saving frames to {NEW_IMAGES_MEMMAP_PATH}")

            frame_count = int(CAPTURE_DURATION * FPS)
            memmap_shape = (frame_count, HEIGHT, WIDTH, 3)
            new_images_memmap = np.memmap(NEW_IMAGES_MEMMAP_PATH, dtype='uint8', mode='w+', shape=memmap_shape)

            for frame_num in range(frame_count):
                frame = picam2.capture_array()
                new_images_memmap[frame_num] = frame

            new_images_memmap.flush()

            composites_paths = list(reversed(sorted([os.path.join(PLAY_DIR, f) for f in os.listdir(PLAY_DIR)])))
            most_recent_memmap_composite_path = composites_paths[0]
            most_recent_composite_memmap = np.memmap(most_recent_memmap_composite_path, dtype='uint8', mode='r', shape=memmap_shape)

            output_memmap_path = os.path.join(PLAY_DIR, f"{current_time}.dat")
            t = datetime.now().strftime("%H-%M-%S")
            print(f"{t} - Combining frames from {NEW_IMAGES_MEMMAP_PATH} and {most_recent_memmap_composite_path} to create {output_memmap_path}")

            output_memmap = np.memmap(output_memmap_path, dtype='uint8', mode='w+', shape=memmap_shape)

            expected_shape = input_details[0]['shape']  # Get the expected shape
            _, target_height, target_width, _ = expected_shape  # Extract the expected height and width

            for frame_num in range(frame_count):
                # Resize the input frames to match the expected size
                resized_frame_1 = cv2.resize(new_images_memmap[frame_num], (target_width, target_height))
                resized_frame_2 = cv2.resize(most_recent_composite_memmap[frame_num], (target_width, target_height))

                # Prepare the input tensors
                input_1 = np.expand_dims(resized_frame_1, axis=0).astype(np.float32) / 255.0
                input_2 = np.expand_dims(resized_frame_2, axis=0).astype(np.float32) / 255.0

                interpreter.set_tensor(input_details[0]['index'], input_1)
                interpreter.set_tensor(input_details[1]['index'], input_2)

                # Run inference
                interpreter.invoke()

                # Get the result
                output_frame = interpreter.get_tensor(output_details[0]['index'])

                # Debug: Print the output tensor values
                print(f"Frame {frame_num}: Output tensor min={output_frame.min()}, max={output_frame.max()}")

                # Scale the output back to 0-255 range and convert to uint8
                output_memmap[frame_num] = (output_frame[0] * 255).astype(np.uint8)


            output_memmap.flush()

            del new_images_memmap, most_recent_composite_memmap, output_memmap

            # Clean up old files from play dir if there are too many
            if len(composites_paths) > 5:
                for f in composites_paths[5:]:
                    os.remove(f)
        else:
            print(f"No face detected: {current_time}")
            time.sleep(1)

        print("--------------------------------------------")
