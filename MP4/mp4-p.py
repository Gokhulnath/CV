import cv2
import numpy as np

# p1

image = np.ones((500, 500, 3), dtype=np.uint8) * 255
cv2.circle(image, (100, 40), 10, (0, 0, 255), -1)[1][3]
angle = np.radians(60)
rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
a = np.array([100, 40])
b = np.dot(rotation_matrix, a).astype(int)
cv2.circle(image, tuple(b), 10, (0, 255, 0), -1)[1][3]
cv2.circle(image, (100, 100), 10, (0, 0, 0), -1)[1][3]
c = np.array([100, 100])
ac = a - c
rotated_ac = np.dot(rotation_matrix, ac)
d = (rotated_ac + c).astype(int)
cv2.circle(image, tuple(d), 10, (255, 0, 0), -1)[1][3]

cv2.imshow('Geometric Transformations', image)
cv2.imwrite("p1.png",image)

# p2

img = cv2.imread('lena.png')
rows, cols = img.shape[:2]

M_translate = np.float32([
    [1, 0, 100],
    [0, 1, 200]
])
translated_img = cv2.warpAffine(img, M_translate, (cols, rows))

M_flip = np.float32([
    [-1, 0, cols - 1],
    [0, 1, 0]
])
flipped_img = cv2.warpAffine(img, M_flip, (cols, rows))

angle = np.radians(45)
M_rotate_origin = np.float32([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0]
])
rotated_origin_img = cv2.warpAffine(img, M_rotate_origin, (cols, rows))

center = (cols / 2, rows / 2)
M_rotate_center = np.float32([
    [np.cos(angle), -np.sin(angle), center[0] * (1 - np.cos(angle)) + center[1] * np.sin(angle)],
    [np.sin(angle), np.cos(angle), center[1] * (1 - np.cos(angle)) - center[0] * np.sin(angle)]
])
rotated_center_img = cv2.warpAffine(img, M_rotate_center, (cols, rows))

cv2.imshow('Original', img)
cv2.imshow('Translated', translated_img)
cv2.imwrite("Translated.png",translated_img)
cv2.imshow('Flipped', flipped_img)
cv2.imwrite("Flipped.png",flipped_img)
cv2.imshow('Rotated around origin', rotated_origin_img)
cv2.imwrite("Rotated_around_origin.png",rotated_origin_img)
cv2.imshow('Rotated around center', rotated_center_img)
cv2.imwrite("rotated_center_img.png",rotated_center_img)
cv2.waitKey(0)
cv2.destroyAllWindows()