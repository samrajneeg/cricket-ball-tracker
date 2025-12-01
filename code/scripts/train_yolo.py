import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='code/scripts/ball-dataset.yaml',
help='path to YAML dataset file for ultralytics')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--model', default='yolov8n.pt')
parser.add_argument('--save_dir', default='models')
args = parser.parse_args()

model = YOLO(args.model)

# Fit will create save dir by default under runs/train
model.train(data=args.data, epochs=args.epochs, project=args.save_dir, name='yolov8-ball')
print('Training finished — check', args.save_dir)