import argparse
import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
import pandas as pd
from utils.tracker import Tracker
from utils.contours import Labels

parser = argparse.ArgumentParser()
parser.add_argument('--weights')
parser.add_argument('--source', required=True, help='input video')
parser.add_argument('--type', default='contour')
parser.add_argument('--ball_colour', required=True)
parser.add_argument('--out', default='results/output.mp4')
parser.add_argument('--ann', default='results/ann.json')
parser.add_argument('--conf', type=float, default=0.25)
parser.add_argument('--iou', type=float, default=0.45)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
os.makedirs(os.path.dirname(args.ann), exist_ok=True)

if args.type=='model':
    model = YOLO(args.weights)
    tracker = Tracker(max_distance=80, max_age=15, n_init=1)

    cap = cv2.VideoCapture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    frame_idx = 0
    ann = []
    trajectory = []


    while True:
        ret, frame = cap.read()
        if not ret: break
        results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
        # results is a list; we take first
        dets = []
        r = results[0]
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy() # [x1,y1,x2,y2]
                x1,y1,x2,y2 = xyxy
                cx = float((x1+x2)/2.0)
                cy = float((y1+y2)/2.0)
                dets.append([cx, cy])
        tracker.predict()
        tracker.update(dets)
        active = tracker.get_active_tracks()

        chosen = None
        if len(active) > 0:
        # pick track with is_confirmed==1 and smallest id (deterministic)
            confirmed = [a for a in active if a[3]==1]
            if len(confirmed) == 0:
                chosen = active[0]
            else:
                chosen = sorted(confirmed, key=lambda x: x[0])[0]
        if chosen is None:
            visibility = 0
            cx,cy = -1,-1
        else:
            visibility = chosen[3]
            cx,cy = chosen[1], chosen[2]
            trajectory.append((int(cx), int(cy)))

        ann.append({'frame_index': frame_idx, 'x_centroid': cx, 'y_centroid': cy, 'visibility_flag': int(visibility)})
        # draw overlays
        out_frame = frame.copy()
        # draw all active tracks
        for tid,x,y,conf in active:
            color = (0,255,0) if conf==1 else (0,128,255)
            cv2.circle(out_frame, (int(x),int(y)), 4, color, -1)
            cv2.putText(out_frame, f"ID:{tid}", (int(x)+5,int(y)-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # draw chosen centroid and trajectory
        if visibility==1:
            cv2.circle(out_frame, (int(cx),int(cy)), 6, (0,0,255), -1)
        for i in range(1, len(trajectory)):
            cv2.line(out_frame, trajectory[i-1], trajectory[i], (255,0,0), 2)
        
        out_vid.write(out_frame)
        frame_idx += 1

    cap.release()
    out_vid.release()

    with open(args.ann, 'w') as f:
        json.dump(ann, f, indent=2)

    pd.DataFrame(ann).to_csv(args.ann.replace('.json','.csv'), index=False)
    print('Wrote', args.ann)
    print('Wrote', args.ann.replace('.json','.csv'))

else:
    labels = Labels(args.source, args.ball_colour)
    result_images = labels.predict()
    ann = []
    cap = cv2.VideoCapture(args.source)
    success, frame = cap.read()
    height, width = frame.shape[:2]
    frame = frame[int(height/5):int(4*height/5), 0:width]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out, fourcc, fps, (frame.shape[1], frame.shape[0]))
    cap.release()

    for item in result_images:
        image, frame, x_centroid, y_centroid, visibility = item
        ann.append({'frame_index': frame, 'x_centroid': x_centroid, 'y_centroid': y_centroid, 'visibility_flag': int(visibility)})
        # Writing to an image array
        out.write(image)

    with open(args.ann, 'w') as f:
        json.dump(ann, f, indent=2)
    out.release()