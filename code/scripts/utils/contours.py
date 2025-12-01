import argparse
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import re


class Labels:
    def __init__(self, file_path, ball_type):
        self.file = file_path
        self.ball = ball_type

    def predict(self):
        cap = cv2.VideoCapture(self.file)
        frame_idx = 0
        success, frame = cap.read()
        frames = []
        while success:
            #frame = cv2.resize(frame, (0,0))
            height, width = frame.shape[:2]
            #print(height, width)
            frame = frame[int(height/5):int(4*height/5), 0:width]
            frames.append(frame)
            frame_idx += 1
            success, frame = cap.read()
        cap.release()

        images = []
        trajectory_points = []
        radius_points = []
        width_height = []
        result_images = []

# Loop through frames
        for i, img in enumerate(frames):
            height, width = img.shape[:2]
            #result = img.copy()

            # Convert image to HSV
            image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(image, np.array([0, 0, 160]), np.array([180, 60, 255]))
            if self.ball == 'red':
                lower_red1 = np.array([0, 85, 35])
                upper_red1 = np.array([10, 255, 255])

                lower_red2 = np.array([170, 85, 35])
                upper_red2 = np.array([180, 255, 255])

                # Combine masks for red hue wrapping
                mask1 = cv2.inRange(image, lower_red1, upper_red1)
                mask2 = cv2.inRange(image, lower_red2, upper_red2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower_white = np.array([0, 0, 160])
                upper_white = np.array([180, 60, 255])
                mask = cv2.inRange(image, lower_white, upper_white)

            # Remove noise
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # --------------------------
            # 2. Find contours
            # --------------------------
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contour_image = np.zeros_like(img) 
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            detectedCircles = []
            image = img.copy()

            # Loop through contours and detect circles
            for j, c in enumerate(contours):
                blobArea = cv2.contourArea(c)
                blobPerimeter = cv2.arcLength(c, True)

                if blobPerimeter != 0:
                    blobCircularity = (4 * 3.1416 * blobArea) / (blobPerimeter**2)
                    minCircularity = 0.4
                    minArea = [15, 55]

                    # Get enclosing circle
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    center = (int(x), int(y))
                    radius = int(radius)

                    # Check circularity and area conditions
                    if blobCircularity > minCircularity and blobArea>minArea[0]:
                        detectedCircles.append([center, radius, blobArea, blobCircularity])

            # Process detected circles
            if len(detectedCircles) != 0:
                smallest_blob = max(detectedCircles, key=lambda x: x[3])
                smallest_center, smallest_radius, smallest_area, smallest_circularity = smallest_blob

                w, h = 4 * smallest_radius, 4 * smallest_radius
                color = (255, 0, 0)

                x, y = smallest_center
                cv2.circle(image, smallest_center, smallest_radius, color, 2)
                cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 0, 255), 3)
                trajectory_points.append(smallest_center)
                radius_points.append(smallest_radius)
                width_height.append([w, h])
                result_images.append([image, i, x, y, 1])
            else:
                result_images.append([image, i, -1, -1, 0])
        
        return result_images