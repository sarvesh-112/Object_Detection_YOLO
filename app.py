from flask import Flask, request, render_template, url_for
import os
import cv2
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
MODEL_PATH = 'runs_advanced/fused_sf_yolov8s2/weights/best.pt'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', user_image=None, count=None, boxes=[])

        # Save uploaded image
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Run detection
        results = model(file_path)
        result_img = results[0].plot()
        result_file_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_file_path, result_img)

        # Extract boxes
        boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append({
                'class': 'person' if cls_id == 0 else f'class_{cls_id}',
                'confidence': round(conf, 3),
                'box': [x1, y1, x2, y2]
            })

        # Count persons
        person_count = sum(1 for box in boxes if box['class'] == 'person')

        image_url = url_for('static', filename=f'results/{filename}')
        return render_template('index.html', user_image=image_url, count=person_count, boxes=boxes)

    return render_template('index.html', user_image=None, count=None, boxes=[])

if __name__ == '__main__':
    app.run(debug=True)
