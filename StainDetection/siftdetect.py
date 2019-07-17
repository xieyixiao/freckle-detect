# -*- coding: utf-8 -*-
""" Created on Sat Jun 11 20:22:51 2016 @author: season """

import cv2
import numpy as np

filename = r'/home/jackxie/文档/PycharmProjects/Pic/7src.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.namedWindow('dst', 0)
cv2.resizeWindow('dst', 1280, 720)
cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
exit(0)