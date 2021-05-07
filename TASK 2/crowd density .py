#!/usr/bin/env python
# coding: utf-8

# In[17]:


get_ipython().system('pip install imutils')


# In[27]:


import cv2
import numpy as np
import imutils

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

things = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


# In[31]:


def main():
    img = cv2.imread('people2.jpg')
    img = imutils.resize(img, width=600)

    (H, W) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.007843, (W, H), 127.5)

    detector.setInput(blob)
    img_detect = detector.forward()

    for i in np.arange(0, img_detect.shape[2]):
        confidence = img_detect[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(img_detect[0, 0, i, 1])

            if things[idx] != "person":
                continue

            img_box = img_detect [0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = img_box.astype("int")

            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)

    cv2.imshow("Results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()


# In[ ]:




