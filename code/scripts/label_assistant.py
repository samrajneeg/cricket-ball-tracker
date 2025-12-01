import os
import argparse
import glob
import pandas as pd
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--labels_dir', default='data/labels')
parser.add_argument('--frames_dir', default='data/frames')
parser.add_argument('--out_csv', default='data/annotations/centroids.csv')
args = parser.parse_args()


os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
label_files = sorted(glob.glob(os.path.join(args.labels_dir, '*.txt')))
rows = []
for lf in label_files:
    base = os.path.splitext(os.path.basename(lf))[0]
    frame_path = os.path.join(args.frames_dir, f"{base}.jpg")
    if not os.path.exists(frame_path):
        continue
    w,h = Image.open(frame_path).size
    with open(lf,'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls, xc, yc, bw, bh = parts[:5]
            xc = float(xc) * w
            yc = float(yc) * h
            rows.append({'frame_index': int(base), 'x_centroid': xc, 'y_centroid': yc, 'visibility_flag': 1})


pd.DataFrame(rows).to_csv(args.out_csv, index=False)
print('Wrote', args.out_csv)