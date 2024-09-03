from typing import List, Tuple
import cv2
from cv2 import VideoCapture
from numpy import ndarray
from ultralytics import YOLO

import numpy as np
# fourcc = cv2.VideoWriter_fourcc()  # Codec for MP4 format

class PersonAttireProcessor:
    """
    A class for detecting people in video frames and checking for inappropriate dressing (nudity).
    The class loads YOLOv8 models for person detection and nudity detection, processes images,
    and applies blurring to detected regions where nudity is identified.
    """

    def __init__(self, yolov8_model_path: str, nudity_model_path: str) -> None:
        """
        Initialize the processor with model paths.

        Args:
            yolov8_model_path (str): Path to the YOLOv8 model weights file for person detection.
            nudity_model_path (str): Path to the YOLOv8 model weights file for nudity detection.
        """
        self.yolov8_model = YOLO(yolov8_model_path)
        self.nudity_model = YOLO(nudity_model_path)

    def setup_video_capture(self, video_path: str) -> VideoCapture:
        """
        Initialize video capture from a file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            VideoCapture: A video capture object for reading frames from the video file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video {video_path}.")
        return cap

    def blur_nudity_region(self, frame: ndarray, x1: int, y1: int, x2: int, y2: int, blur_ratio: int) -> ndarray:
        """
        Apply a blur effect to a detected region with nudity in the video frame.

        Args:
            frame (ndarray): The video frame containing the detected region.
            x1 (int): Top-left x-coordinate of the bounding box.
            y1 (int): Top-left y-coordinate of the bounding box.
            x2 (int): Bottom-right x-coordinate of the bounding box.
            y2 (int): Bottom-right y-coordinate of the bounding box.
            blur_ratio (int): Ratio for blurring the detected region.

        Returns:
            ndarray: Video frame with the nudity region blurred.
        """
        nudity_roi = frame[y1:y2, x1:x2]
        blurred_nudity = cv2.blur(nudity_roi, (blur_ratio, blur_ratio))  # Adjust the kernel size for blur
        frame[y1:y2, x1:x2] = blurred_nudity
        return frame

    def process_frame(self, frame: ndarray, person_conf: float, nudity_conf: float, blur_ratio: int) -> ndarray:
        """
        Detect people in a video frame and evaluate their attire for nudity. Blur areas with nudity.

        Args:
            frame (ndarray): Input video frame.
            person_conf (float): Confidence threshold for person detections.
            nudity_conf (float): Confidence threshold for nudity detection.
            blur_ratio (int): Ratio for blurring nudity regions.

        Returns:
            ndarray: Video frame with detected people labeled and nudity blurred.
        """
        person_results = self.yolov8_model(frame)

        for result in person_results:
            for i, bbox in enumerate(result.boxes):
                if result.names[int(result.boxes.cls[i])] == 'person':  # Assuming 'person' is the label
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                    # Draw the bounding box around the person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label_size, base_line = cv2.getTextSize("person", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # y1 = max(y1, label_size[1] + 100)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1 + base_line - 10),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, "person", (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Extract the person region (ROI)
                    person_roi = frame[y1:y2, x1:x2]

                    # Nudity detection
                    nudity_results = self.nudity_model.predict(person_roi, show=False)

                    for j, nudity_bbox in enumerate(nudity_results[0].boxes):
                        nudity_confidence = nudity_results[0].boxes.conf[j].item()  # Get confidence

                        if nudity_confidence >= nudity_conf:  # Check if the confidence is above the threshold
                            nudity_x1, nudity_y1, nudity_x2, nudity_y2 = map(int, nudity_bbox.xyxy[0])  # Get the detected nudity box

                            # Adjust coordinates to the full frame
                            nudity_x1 += x1
                            nudity_x2 += x1
                            nudity_y1 += y1
                            nudity_y2 += y1

                            # Apply blur to the nudity region
                            frame = self.blur_nudity_region(frame, nudity_x1, nudity_y1, nudity_x2, nudity_y2, blur_ratio)

        return frame

    def run(self, video_path: str,output_file_name: str, person_conf: float = 0.5, nudity_conf: float = 0.1, blur_ratio: int = 50) -> None:
        """
        Execute the person detection, nudity detection, labeling, and blurring process on a video file.

        Args:
            video_path (str): Path to the video file to be processed.
            person_conf (float): Confidence threshold for person detection.
            nudity_conf (float): Confidence threshold for nudity detection.
            blur_ratio (int): Ratio for blurring the detected nudity regions.
        """
        cap = self.setup_video_capture(video_path)
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            processed_frame = self.process_frame(frame, person_conf, nudity_conf, blur_ratio)
            out.write(processed_frame)
            cv2.imshow("Video Frame", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    yolov8_model_path = "yolov8s.pt"
    nudity_model_path = "N_300 (1).pt"

    processor1 = PersonAttireProcessor(yolov8_model_path, nudity_model_path)
    processor1.run(video_path="Untitled.mp4",
                   output_file_name="Untitledout.mp4", blur_ratio=20)

    processor2 = PersonAttireProcessor(yolov8_model_path, nudity_model_path)
    processor2.run(video_path="🇦🇷 Villa Gesell Beach Walk Perfect Day at Argentina.mp4",output_file_name="Villa_out1.mp4",blur_ratio=20)


    processor = PersonAttireProcessor(yolov8_model_path, nudity_model_path)
    processor.run(video_path="🇦🇷 Mar Del Plata Beach Walk Hot Day at Argentina.mp4",output_file_name="marout1.mp4",blur_ratio=20)
