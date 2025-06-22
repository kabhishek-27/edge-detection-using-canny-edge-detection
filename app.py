from tempfile import template
from flask import Flask, render_template, request, jsonify
import numpy as np
from io import BytesIO
from flask_cors import CORS
from lib.cannyEdgeDetectionAlgo import apply_canny_edge_detection


app = Flask(__name__)
CORS(app,resources={r"/upload": {"origins": "*"}})
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        image_bytes = file.read()
        result_bytes = apply_canny_edge_detection(image_bytes)
        return (
            result_bytes,
            200,
            {'Content-Type': 'image/png'}
        )
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    
    app.run(debug=True)