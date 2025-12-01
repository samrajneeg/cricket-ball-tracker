import cv2
import os
import argparse
import re
from pathlib import Path
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--out_dir', default='data/frames')
parser.add_argument('--resize', type=float, default=1.0)
args = parser.parse_args()

video_files = []
# Common video extensions
extensions = ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.wmv') 
for ext in extensions:
    # Construct the full pattern for the glob search
    ip = args.input.split('/')
    pattern = os.path.join(ip[0], ip[1], ext)
    # Use glob.glob() to find all files matching the pattern
    for file in glob.glob(pattern):
        video_files.append(file)

video_files = sorted(video_files, key= lambda x: int(str(x.split('\\')[-1]).split('.')[0]))
#print(video_files)
for file in video_files:
    os.makedirs(args.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(file)
    frame_idx = 0
    success, frame = cap.read()
    while success:
        frame = cv2.resize(frame, (0,0), fx=args.resize, fy=args.resize)
        height, width = frame.shape[:2]
        #print(height, width)
        frame = frame[int(height/5):int(4*height/5), 0:width]
        fname = os.path.join(args.out_dir, f"{file.split('\\')[-1].split('.')[0]}_{frame_idx:06d}.jpg")
        cv2.imwrite(fname, frame)
        frame_idx += 1
        success, frame = cap.read()
    cap.release()
    print('Extracted', frame_idx, 'frames to', args.out_dir)
