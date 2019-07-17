# coding=utf-8

import cv2
import dlib
import numpy as np

path = r".\pic\2src.jpg"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 人脸分类器
detector = dlib.get_frontal_face_detector()
# 获取人脸检测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
dets = detector(gray, 0)
for face in dets:
    landmarks = np.array([[p.x, p.y] for p in predictor(img, face).parts()])
    # 遍历所有点，打印出其坐标，并圈出来
    for idx, pt in enumerate(landmarks):
        pt_pos = (pt[0], pt[1])
        cv2.circle(img, pt_pos, 10, (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx + 1), pt_pos, font, 2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.namedWindow("image", 0)
    cv2.resizeWindow("image", 720, 1800)
    cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
