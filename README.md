# cricket-ball-tracker
<h1>Setup & Usage</h1>

<p>Follow these steps to run the project. The data should be present in the data/raw folder with the data folder being at the same level as the code folder.</p>

<ol>
  <li>
    <strong>Install dependencies</strong>
    <div style="margin-top:8px;">
      <pre style="background:#f6f8fa;border:1px solid #d1d5da;padding:12px;border-radius:6px;overflow:auto;">
<code>pip install -r requirements.txt</code>
      </pre>
    </div>
  </li>

  <li style="margin-top:16px;">
    <strong>Extract frames</strong>
    <div style="margin-top:8px;">
      <pre style="background:#f6f8fa;border:1px solid #d1d5da;padding:12px;border-radius:6px;overflow:auto;">
<code>python code/scripts/extract_frames.py --input data/raw/match.mp4</code>
      </pre>
    </div>
  </li>

  <li style="margin-top:16px;">
    <strong>Prepare labels for YOLOv8 training</strong>
    <div style="margin-top:8px;">
      <pre style="background:#f6f8fa;border:1px solid #d1d5da;padding:12px;border-radius:6px;overflow:auto;">
<code>python code/scripts/yolo_label.py</code>
      </pre>
    </div>
  </li>

  <li style="margin-top:16px;">
    <strong>Train model (on Colab)</strong>
    <div style="margin-top:8px;">
      <pre style="background:#f6f8fa;border:1px solid #d1d5da;padding:12px;border-radius:6px;overflow:auto;">
<code>!cd "/content/drive/My Drive/code/scripts"
!python "/content/drive/My Drive/code/scripts/train_yolo.py" --data "/content/drive/My Drive/code/scripts/ball-dataset.yaml" --epochs 50 --save_dir "/content/drive/My Drive/models"</code>
      </pre>
    </div>
  </li>

  <li style="margin-top:16px;">
    <strong>Detect & track</strong>
    <div style="margin-top:8px;">
      <pre style="background:#f6f8fa;border:1px solid #d1d5da;padding:12px;border-radius:6px;overflow:auto;">
<code>python code/scripts/detect_track.py --weights models/best_ball.pt --source data/raw/2.mov --out results/model/2.mov --ann annotations/model/ann_2.json --ball_colour red</code>
      </pre>
    </div>
  </li>
</ol>
