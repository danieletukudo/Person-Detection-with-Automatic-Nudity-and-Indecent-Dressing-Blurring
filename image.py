from typing import List, Tuple
import cv2
from ultralytics import YOLO
import numpy as np

class ImageProcessor:
    """
    A class for processing images to detect people and check for inappropriate dressing (nudity) using YOLOv8 models.
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
        self.person_model = YOLO(yolov8_model_path)
        self.nudity_model = YOLO(nudity_model_path)

    def blur_nudity(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, blur_ratio: int) -> np.ndarray:
        """
        Apply a blur effect to a detected region with nudity in the image.

        Args:
            frame (np.ndarray): The image containing the detected nudity.
            x1 (int): Top-left x-coordinate of the nudity bounding box.
            y1 (int): Top-left y-coordinate of the nudity bounding box.
            x2 (int): Bottom-right x-coordinate of the nudity bounding box.
            y2 (int): Bottom-right y-coordinate of the nudity bounding box.
            blur_ratio (int): Ratio for blurring the nudity region.

        Returns:
            np.ndarray: Image with the nudity region blurred.
        """
        nudity_roi = frame[y1:y2, x1:x2]
        blur_nudity = cv2.blur(nudity_roi, (blur_ratio, blur_ratio))  # Adjust the kernel size for blur
        frame[y1:y2, x1:x2] = blur_nudity
        return frame

    def process_image(self, image_path: str, blur_ratio: int = 50) -> None:
        """
        Detect people and evaluate their attire for appropriateness in an image. Blur areas with inappropriate attire.

        Args:
            image_path (str): Path to the image file.
            blur_ratio (int): Ratio for blurring inappropriate attire regions.
        """
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error: Could not read image {image_path}.")
            return

        # Detect people in the image using the person detection model
        person_results = self.person_model(frame)

        for result in person_results:
            for i, bbox in enumerate(result.boxes):
                if result.names[int(result.boxes.cls[i])] == 'person':  # Assuming 'person' is the label
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255),
                                  2)  # Red for nudity

                    # Add the label "Person" above the bounding box
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

                        if nudity_confidence >= 0.3:  # Check if the confidence is above the threshold
                            nudity_x1, nudity_y1, nudity_x2, nudity_y2 = map(int, nudity_bbox.xyxy[0])  # Get the detected nudity box

                            # Adjust coordinates to the full frame
                            nudity_x1 += x1
                            nudity_x2 += x1
                            nudity_y1 += y1
                            nudity_y2 += y1

                            # Apply blur to the nudity region
                            frame = self.blur_nudity(frame, nudity_x1, nudity_y1, nudity_x2, nudity_y2, blur_ratio)

                            # Optional: Draw bounding box around the blurred area
                            # cv2.rectangle(frame, (nudity_x1, nudity_y1), (nudity_x2, nudity_y2), (0, 0, 255), 2)  # Red for nudity

        # Show the image with bounding boxes
        cv2.imshow('Processed Output', frame)
        print(f"Processed image: {image_path}")

        # Wait for a key press and then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_file = '../img_8.png'
    yolov8_model_path = 'yolov8s.pt'
    nudity_model_path = '../N_300 (1).pt'

    processor = ImageProcessor(yolov8_model_path, nudity_model_path)
    processor.process_image(image_path=image_file,blur_ratio=20)

