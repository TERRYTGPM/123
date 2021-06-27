import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image 
import PIL.ImageOps
import cv2
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
Y = pd.read_csv('dog.csv')['labels']
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','Z']
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=9, train_size=7500, test_size=2500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upper_left = (int(width / 2 - 20), int(height / 2 - 20))
        bottom_right = (int(width / 2 + 20), int(height / 2 + 20))

        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        im_pil = Image.fromarray(roi)

        image_bw = im_pil.convert("L")

        ri = image_bw.resize((28, 28), Image.ANTIALIAS)
        rii = PIL.ImageOps.invert(ri)

        pf = 20

        minpx = np.percentile(rii, pf)

        riis = np.clip(rii-minpx, 0, 255)

        maxpx = np.max(rii)
        riis = np.asarray(riis)/maxpx

        testsample = np.array(riis).reshape(1, 784)
        pred = clf.predict(testsample)
        print(pred)
        cv2.imshow("f", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.DestroyAllWindows()