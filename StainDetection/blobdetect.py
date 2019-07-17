import cv2
import numpy as np

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 30
params.maxThreshold = 300
# Filter by Area.
params.filterByArea = True
params.minArea = 15
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.3
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
# Create a detector with the parameters
ver = cv2.__version__.split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)
# Read image
image = cv2.imread(r".\Pic\2src.jpg")
im_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
im = cv2.equalizeHist(im_1)
# Set up the detector with default parameters.
# detector = cv2.SimpleBlobDetector_create()
# Detect blobs.
keypoints = detector.detect(im)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Show keypoints
cv2.namedWindow("image", 0)
cv2.resizeWindow("image", 720, 1800)
cv2.imshow("image", im_with_keypoints)
# cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)


