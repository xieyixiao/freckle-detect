# -*- coding: UTF-8 -*-

"""
opencv实现人脸识别
参考：
1、https://github.com/opencv/opencv/tree/master/data/haarcascades
2、http://www.cnblogs.com/hanson1/p/7105265.html

"""

import cv2

# 待检测的图片路径
# from cv2 import CascadeClassifier

imagepath = r"C:\Users\xieyi\Desktop\Newfolder\Pic\tx1502.JPG"

image = cv2.imread(imagepath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# "参考：https://github.com/opencv/opencv/tree/master/data/haarcascades\n")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#探测人脸
"""
# 根据训练的数据来对新图片进行识别的过程。
# 1. detectMultiScale function是一个检测对象的通用函数。由于我们调用它检测人脸，所以它就将检测人脸。第一个选项是图片灰度转换的相片。
# 2. 第二个选项是scaleFactor。由于可能存在部分人脸距离相机相对较近，从而与较远处的人脸相比，尺寸较大。比例因子可以进行抵消补偿。
# 3. 检测算法使用了一个活动窗口进行目标检测。minNeighbors选项定义了在其声明人脸被找到前，当前对象附件的目标数量。同时，minSize选项给出了每个窗口的大小。
"""
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=10,
    minSize=(10, 10),
    # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)

# 我们可以随意的指定里面参数的值，来达到不同精度下的识别。返回值就是opencv对图片的探测结果的体现。

# 处理人脸探测的结果
print("发现{0}个人脸!".format(len(faces)))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + w), (0, 255, 0), 2)
    # cv2.circle(image,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)
cv2.namedWindow("image", 0)
cv2.resizeWindow("image", 1280, 720)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)