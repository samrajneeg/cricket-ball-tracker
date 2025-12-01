# cricket-ball-tracker
1. Install dependencies
pip install -r requirements.txt
2. Extract frames
python code/scripts/extract_frames.py --input data/raw/match.mp4
3. Prepare labels for YOLOv8 training
python code/scripts/yolo_label.py
4. Train model (on Colab)
!cd "/content/drive/My Drive/code/scripts"
!python "/content/drive/My Drive/code/scripts/train_yolo.py" --data "/content/drive/My Drive/code/scripts/ball-dataset.yaml" --epochs 10 --save_dir "/content/drive/My Drive/models"
6. Detect track
python code/scripts/detect_track.py --weights models/best_ball.pt --source data/raw/2.mov --out results/model/2.mov --ann annotations/model/ann_2.json --ball_colour red  
