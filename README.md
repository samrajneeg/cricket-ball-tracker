# cricket-ball-tracker
1. Install dependencies
<br>
pip install -r requirements.txt
<br>
2. Extract frames
<br>
python code/scripts/extract_frames.py --input data/raw/match.mp4
3. Prepare labels for YOLOv8 training
<br>
python code/scripts/yolo_label.py
4. Train model (on Colab)
<br>
!cd "/content/drive/My Drive/code/scripts"
<br>
!python "/content/drive/My Drive/code/scripts/train_yolo.py" --data "/content/drive/My Drive/code/scripts/ball-dataset.yaml" --epochs 10 --save_dir "/content/drive/My Drive/models"
5. Detect track
<br>
python code/scripts/detect_track.py --weights models/best_ball.pt --source data/raw/2.mov --out results/model/2.mov --ann annotations/model/ann_2.json --ball_colour red  
