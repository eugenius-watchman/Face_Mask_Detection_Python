import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load the images
with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

# Ensure the arrays are not empty
assert with_mask.size > 0, "with_mask.npy is empty"
assert without_mask.size > 0, "without_mask.npy is empty"

# Check the shapes of the images
print("Shape of 'with_mask' image array:", with_mask.shape)
print("Shape of 'without_mask' image array:", without_mask.shape)

# Reshape the images if necessary
if with_mask.ndim > 2:  # if the images are not already flattened
    with_mask = with_mask.reshape(with_mask.shape[0], -1)
if without_mask.ndim > 2:  # if the images are not already flattened
    without_mask = without_mask.reshape(without_mask.shape[0], -1)

print("Shape of 'with_mask' after reshaping:", with_mask.shape)
print("Shape of 'without_mask' after reshaping:", without_mask.shape)

# Combine the data and prepare training data
X = np.r_[with_mask, without_mask]
print("Combined data shape:", X.shape)

# Create labels
labels = np.zeros(X.shape[0])
labels[len(with_mask):] = 1.0  # Assuming 0 for with_mask and 1 for without_mask

names = {0: 'Mask', 1: 'No Mask'}

# Perform ML Algorithm
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.20, random_state=42)

# Dimensionality reduction
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print("Shape of training data after PCA:", x_train.shape)
print("Shape of test data after PCA:", x_test.shape)

# Apply SVM
svm = SVC()
svm.fit(x_train, y_train)

# Predict and calculate accuracy
y_pred = svm.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

haar_data = cv2.CascadeClassifier('haar_data.xml')
capture = cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:
        face = haar_data.detectMultiScale(img)
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face_rect = img[y:y + h, x:x + w, :]
            face_rect = cv2.resize(face_rect, (50, 50))
            face_rect = face_rect.reshape(1, -1)
            face_rect = pca.transform(face_rect)  # Apply PCA transformation
            pred = svm.predict(face_rect)
            n = names[int(pred)]
            print(n)
            # if n != 0 and n != 1:
            # print('Improper Mask')
        cv2.imshow('result', img)
    if cv2.waitKey(2) == 27:
        break

# Release the capture and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()
