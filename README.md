# edge-detection-using-canny-edge-detection

This is a simple Flask web application that allows users to upload an image and applies the Canny Edge Detection algorithm to it. The processed image is then returned and displayed on the same page.

## 📁 Project Structure

├── app.py # Main Flask application
├── cannyEdgeDetectionAlgo.py # Canny edge detection logic (OpenCV)
├── templates/
│ └── index.html # Frontend for image upload
└── README.md # Project documentation

## 🚀 Features

- Upload any image using a clean HTML form.
- Preview the original image before submitting.
- Process the uploaded image on the server using Canny Edge Detection.
- Return and display the processed image dynamically.

Note: 
- python -m venv venv
- source venv/bin/activate

Install all dependencies with:
- pip install -r requirements.txt
