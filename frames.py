import cv2
import os
import subprocess
import time
import mediapipe as mp
import numpy as np



WIDTH = 1080 #720
HEIGHT = 1920 # 1280


def get_user():
    result = subprocess.run(["whoami"], capture_output=True, text=True, check=True)
    return result.stdout.strip()

if get_user() == "pi":
    from picamera2 import Picamera2


def save_frames_from_video(camera_index=0, num_chunks=1, chunk_duration=5, output_file="frames.dat", batch_size=10):
    if get_user() == "pi":
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (WIDTH, HEIGHT)}))
        picam2.start()
        fps = 30
    else:
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)

    chunk_frame_count = int(chunk_duration * fps)
    total_frames = num_chunks * chunk_frame_count
    frame_shape = (HEIGHT, WIDTH, 3)
    dtype = np.uint8

    # Memory-map the file for the entire sequence
    mmapped_frames = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_frames,) + frame_shape)

    frame_index = 0

    for chunk in range(num_chunks):
        batch_frames = []

        for frame_num in range(chunk_frame_count):
            print(frame_num)
            if get_user() == "pi":
                frame = picam2.capture_array()
                # Drop the alpha channel (if present) to match RGB format
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            # Store the frame in the batch
            batch_frames.append(frame)

            # Write and flush the batch to disk if the batch size is reached
            if len(batch_frames) == batch_size:
                for bf in batch_frames:
                    mmapped_frames[frame_index] = bf
                    frame_index += 1
                mmapped_frames.flush()
                batch_frames = []  # Reset the batch

            time.sleep(0.01)  # Small delay to reduce load

        # Write and flush any remaining frames in the batch
        if batch_frames:
            for bf in batch_frames:
                mmapped_frames[frame_index] = bf
                frame_index += 1
            mmapped_frames.flush()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if get_user() == "pi":
        picam2.stop()
        picam2.close()
    else:
        cap.release()
        cv2.destroyAllWindows()

    del mmapped_frames  # Ensure the data is flushed to disk


def alpha_blend_images(image1, image2, alpha=0.5):
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)


def overlay_frames_from_memmap(input_file="frames.dat", num_chunks=4, chunk_duration=5, alpha=0.5, output_file="overlay.dat"):
    fps = 30
    chunk_frame_count = int(chunk_duration * fps)
    total_frames = num_chunks * chunk_frame_count
    frame_shape = (HEIGHT, WIDTH, 3)
    dtype = np.uint8

    mmapped_input = np.memmap(input_file, dtype=dtype, mode='r', shape=(total_frames,) + frame_shape)
    mmapped_output = np.memmap(output_file, dtype=dtype, mode='w+', shape=(chunk_frame_count,) + frame_shape)

    for frame_index in range(chunk_frame_count):
        composite_frame = mmapped_input[frame_index]

        for chunk_index in range(1, num_chunks):
            frame = mmapped_input[chunk_index * chunk_frame_count + frame_index]
            composite_frame = alpha_blend_images(composite_frame, frame, alpha)

        mmapped_output[frame_index] = composite_frame

    del mmapped_input
    del mmapped_output


def face_detected_cv():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame from webcam.")
        return False

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30)
    )

    return len(faces) > 0


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def face_detected_mp(confidence_threshold=0.5):
    return True  # Simplified for debugging purposes


def stream_images_from_memmap(output_file="overlay.dat"):
    file_size = os.path.getsize(output_file)
    frame_size = HEIGHT * WIDTH * 3
    num_frames = file_size // frame_size

    mmapped_output = np.memmap(output_file, dtype=np.uint8, mode='r', shape=(num_frames, HEIGHT, WIDTH, 3))

    for frame in mmapped_output:
        cv2.imshow("window", frame)

        key = cv2.waitKey(20)
        if key == ord("q"):
            break

    del mmapped_output


if __name__ == "__main__":
    while True:
        detection = face_detected_mp()
        print(f"Detection Status: {detection}")
        time.sleep(0.5)
