import cv2
from cv2 import imdecode
import numpy as np

position = (100, 10)

img = cv2.imread("test.jpg")
print(img.shape)
img_center = (img.shape[0]//2, img.shape[1]//2)
img = cv2.circle(img, position, 10, (255, 0, 0), -1)
x_shift = img_center[1] - position[1]
y_shift = img_center[0] - position[0]
inds_x = np.arange(img.shape[1])
inds_y = np.arange(img.shape[0])
inds = np.meshgrid(inds_y, inds_x)[::-1]
print(inds[1].shape)
print("Should be 360")
inds[0] -= x_shift
inds[1] -= y_shift

inds[0] %= img.shape[0]
inds[1] %= img.shape[1]

print(img.shape)
img = img[inds]
cv2.imshow("test", img)

cv2.waitKey(0)
cv2.destroyAllWindows()