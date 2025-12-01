import argparse
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import re
import splitfolders

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data/frames')
parser.add_argument('--output', default='data/labels/class0')
args = parser.parse_args()

frames = os.listdir(args.input)
frames.sort(key=lambda f: int(re.sub('_', '', f).split('.')[0]))
video = {'1':'white', '2':'red', '3':'red', '4':'white', '5':'white', '6':'white', 
         '7':'white', '8':'white', '9':'white', '10':'white', '11':'red', 
         '12':'white', '13':'white', '14':'white', '15':'white'}

#frames = frames[31:323]

images = []
trajectory_points = []
radius_points = []
width_height = []

output_directory = args.output
os.makedirs(output_directory, exist_ok=True)

# Loop through frames
for frame_filename in frames:
    frame_path = os.path.join(args.input, frame_filename)
    img = cv2.imread(frame_path)
    height, width = img.shape[:2]
    #print(height, width)
    #img = img[320:height-320, 0:width]
    result = img.copy()

    # Convert image to HSV
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    video_no = frame_filename.split('/')[-1].split('_')[0]

    # Define color ranges for masking
    #lower_red1 = np.array([0, 90, 50])
    #upper_red1 = np.array([10, 255, 255])

    #lower_red2 = np.array([170, 90, 50])
    #upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(image, np.array([0, 0, 160]), np.array([180, 60, 255]))
    if video[video_no] == 'red':
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
    os.makedirs('data/contours', exist_ok=True)
    cv2.imwrite(os.path.join('data/contours', frame_filename), contour_image)

    detectedCircles = []
    image = img.copy()

    # Loop through contours and detect circles
    for i, c in enumerate(contours):
        blobArea = cv2.contourArea(c)
        blobPerimeter = cv2.arcLength(c, True)

        if blobPerimeter != 0:
            blobCircularity = (4 * 3.1416 * blobArea) / (blobPerimeter**2)
            minCircularity = 0.4
            minArea = [15, 55]
            print(frame_filename, i, blobArea, blobCircularity)

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
        images.append((image, frame_filename))
        processed_image = images[-1][0]
        cv2.imwrite(os.path.join(output_directory, images[-1][1]), processed_image)

splitfolders.ratio('data/labels', output="code/scripts/split", seed=42, ratio=(0.8, 0.2))

# Save processed images to 'Frame_b' directory

"""for i in range(len(images)):
    processed_image = images[i][0]
    cv2.imwrite(os.path.join(output_directory, images[i][1]), processed_image)"""

# Create video from processed frames
"""frames = os.listdir('Frame_b/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))
frame_array = []

for i in range(len(frames)):
    # Reading each file
    img = cv2.imread('Frame_b/' + frames[i])
    height, width, layers = img.shape
    size = (width, height)
    # Inserting the frames into an image array
    frame_array.append(img)

# Write video file
out = cv2.VideoWriter('KOHLI_COVER_DRIVE_with_bounding_boxes.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
for i in range(len(frame_array)):
    # Writing to an image array
    out.write(frame_array[i])
out.release() """
