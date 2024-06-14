import cv2
import numpy as np

# Load the Haar Cascade Classifier
haar_data = cv2.CascadeClassifier('haar_data.xml')

# Open the default camera
capture = cv2.VideoCapture(0)

# Initialize lists to hold face data
without_mask_data = []
with_mask_data = []

# Define the maximum number of samples to capture
max_samples = 200


# Function to capture face data
def capture_faces(data_list, label):
    print(f"Starting capture for {label} data...")
    while True:
        flag, img = capture.read()
        if flag:
            face = haar_data.detectMultiScale(img)
            for x, y, w, h in face:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
                face_rect = img[y:y + h, x:x + w, :]
                face_rect = cv2.resize(face_rect, (50, 50))
                print(f"Captured {label} faces:", len(data_list))
                if len(data_list) < max_samples:
                    data_list.append(face_rect)
                cv2.imshow('result', img)
            if cv2.waitKey(2) == 27 or len(data_list) >= max_samples:
                break
    print(f"Completed capture for {label} data.")


# Capture without mask data
capture_faces(without_mask_data, "without mask")

# Release the capture and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()

# Convert without_mask_data to NumPy array and save it
without_mask_data = np.array(without_mask_data)
np.save('without_mask.npy', without_mask_data)
print("Without mask data shape:", without_mask_data.shape)

# Prompt the user to get ready for the with mask data capture
input("Get ready for the 'with mask' data capture and press Enter to continue...")

# Reopen the default camera
capture = cv2.VideoCapture(0)

# Capture with mask data
capture_faces(with_mask_data, "with mask")

# Release the capture and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()

# Convert with_mask_data to NumPy array and save it
with_mask_data = np.array(with_mask_data)
np.save('with_mask.npy', with_mask_data)
print("With mask data shape:", with_mask_data.shape)
