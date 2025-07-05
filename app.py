from flask import Flask, request, render_template, flash
import os
import cv2
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Folder paths
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Mapping for model names and their paths
MODEL_PATHS = {
    'fused_sf': 'runs_advanced/fused_sf_yolov8s2/weights/best.pt',
    'fused_mgf': 'runs_advanced/fused_mgf_yolov8s/weights/best.pt',
    'visible': 'runs_advanced/visible_yolov8s/weights/best.pt',
    'infrared': 'runs_advanced/infrared_yolov8s/weights/best.pt'
}

# Load all models once to avoid reloading on every request
MODELS = {name: YOLO(path) for name, path in MODEL_PATHS.items()}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        model_key = request.form.get('model', 'fused_sf')

        if not file or file.filename == '':
            flash("❌ No file selected.")
            return render_template('index.html', user_image=None, count=None, model_name=model_key, bboxes=[])

        if not allowed_file(file.filename):
            flash("❌ Invalid file type. Please upload a .jpg, .jpeg, or .png image.")
            return render_template('index.html', user_image=None, count=None, model_name=model_key, bboxes=[])

        if model_key not in MODELS:
            flash("❌ Invalid model selection.")
            return render_template('index.html', user_image=None, count=None, model_name='fused_sf', bboxes=[])

        # Save uploaded image
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Perform detection
        model = MODELS[model_key]
        results = model(file_path)

        # Handle case when no detection is made
        if not results or results[0].boxes is None:
            flash("⚠️ No objects detected in the image.")
            return render_template('index.html', user_image=file_path, count=0, model_name=model_key, bboxes=[])

        # Draw and save result image
        result_img = results[0].plot()
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, result_img)

        # Count people and extract bounding boxes
        bboxes = []
        person_count = 0
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_id == 0:
                person_count += 1
            bboxes.append({
                'class': cls_id,
                'confidence': conf,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })

        return render_template(
            'index.html',
            user_image=result_path,
            count=person_count,
            model_name=model_key,
            bboxes=bboxes
        )

    # GET request
    return render_template('index.html', user_image=None, count=None, model_name='fused_sf', bboxes=[])


if __name__ == '__main__':
    app.run(debug=True)
