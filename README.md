A JPG image for experimenting the haar cascade classifier algorithm.
Face Mask Detection using Python OPenCv Machine Learning and Python Sklearn.

### Face Mask Detection System ###

This project aims to detect whether individuals are wearing face masks using a combination of computer vision and machine learning techniques. The system captures facial images from a webcam, processes them, and utilizes a Support Vector Machine (SVM) classifier to predict mask usage.
Features

    Face Data Capture: Captures images of faces with and without masks using Haar Cascade Classifier.
    Data Storage: Saves captured face images as NumPy arrays for training.
    Mask Detection: Uses SVM for classifying faces as "With Mask" or "Without Mask."
    Real-time Detection: Displays the live video feed with predictions overlaid.

Requirements

    OpenCV
    NumPy
    scikit-learn

Installation

    Clone the repository:

    bash

git clone https://github.com/eugenius-watchman/Face_Mask_Detection_Python.git

Navigate to the project directory:

bash

cd restaurant_app

Install the required packages:

bash

    pip install -r requirements.txt

Usage
Step 1: Capture Face Data

To capture images of faces, run the following code snippet:

python

# Import necessary libraries
import cv2
import numpy as np

# Load the Haar Cascade Classifier
haar_data = cv2.CascadeClassifier('haar_data.xml')

# Function to capture face data
def capture_faces(data_list, label):
    ...
    
# Capture without mask data
capture_faces(without_mask_data, "without mask")

# Capture with mask data
capture_faces(with_mask_data, "with mask")

This will prompt you to capture face images without masks first, followed by images with masks.
Step 2: Train the Classifier

After capturing the data, train the SVM classifier using the captured images:

python

# Load the images
with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

# Combine the data and prepare labels
X = np.r_[with_mask, without_mask]
labels = np.zeros(X.shape[0])
labels[len(with_mask):] = 1.0  # 0 for with_mask, 1 for without_mask

# Train the SVM
svm.fit(x_train, y_train)

Step 3: Real-time Mask Detection

To use the trained classifier for real-time mask detection:

python

# Start capturing video
capture = cv2.VideoCapture(0)
while True:
    ...
    pred = svm.predict(face_rect)
    ...
    cv2.imshow('result', img)

Press Esc to exit the live video feed.
Results

The system will display the live video feed with bounding boxes around detected faces, and it will print whether each detected face is wearing a mask or not.
Notes

    Ensure that haar_data.xml is available in the project directory.
    Adjust the max_samples variable to change the number of images captured during training.

Accuracy

The accuracy of the model will be printed after the classification is complete.
License

This project is licensed under the MIT License.
