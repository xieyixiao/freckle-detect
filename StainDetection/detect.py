import cv2
import numpy as np
import dlib
# 读取图像
path = r"./Pic/5src.jpg"
image = cv2.imread(path)
# 人脸分类器
detector = dlib.get_frontal_face_detector()
# 获取人脸检测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 椭圆掩膜
mask = np.zeros(image.shape[0:2], np.uint8)
cv2.ellipse(mask, (int(image.shape[1] / 2-15), int(image.shape[0] / 2+150)),
            (int(image.shape[1] / 2-50), int(image.shape[0] / 2-270)), 0, -180, 180, 255, -1)
# 加掩膜后图像
masked_ori = cv2.bitwise_and(image, image, mask=mask)
masked_dst = masked_ori.copy()
# 灰度处理
im_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# dlib定位五官
dets = detector(im_1, 0)
for face in dets:
    landmarks = np.array([[p.x, p.y] for p in predictor(im_1, face).parts()])
# 直方图均匀化
im_2 = cv2.equalizeHist(im_1)
# 自适应阈值处理
im_3 = cv2.adaptiveThreshold(im_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 211, 3)
# 使用阈值排除五官
im_ex2 = cv2.adaptiveThreshold(im_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 921, 55)
ret, im_ex = cv2.threshold(im_2, 80, 255, cv2.THRESH_BINARY_INV)
im_ex3 = cv2.bitwise_or(im_ex, im_ex2)
# 排除五官后图像
im_4 = cv2.bitwise_or(im_3, im_ex3)
# 形态学处理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
masked_draw = cv2.morphologyEx(im_4, cv2.MORPH_CLOSE, kernel)
# dlib排除五官
cv2.fillPoly(masked_draw, [landmarks[7:11], landmarks[36:42], landmarks[42:48], landmarks[48:60]], 255)
mask = cv2.bitwise_not(mask)
masked_draw = cv2.bitwise_or(masked_draw, mask)
# 绘制色斑轮廓
contours, hierarchy = cv2.findContours(masked_draw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(masked_dst, contours, -1, (0, 255, 0), 3)
# 显示原始图像
cv2.namedWindow("masked_ori", 0)
cv2.resizeWindow("masked_ori", int(3456/8), int(5184/8))
cv2.imshow("masked_ori", masked_ori)
# 显示色斑
cv2.namedWindow("dst", 0)
cv2.resizeWindow("dst", int(3456/8), int(5184/8))
cv2.imshow("dst", masked_dst)
cv2.waitKey(0)


