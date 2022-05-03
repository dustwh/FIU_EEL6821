import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# get bounding box
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


# generate image with black background
img = cv2.imread('demo.png')
rows, cols, dim = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
M = np.float32([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
result = cv2.warpPerspective(img, M, (int(cols), int(rows)))
cv2.imwrite("result.jpg", result)

# projection
result_img = cv2.imread('result.jpg')
result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
rows, cols, dim = result_img.shape
rmin, rmax, cmin, cmax = bbox(result_img)
src = np.array([[cmin,rmin],[cmax,rmin],[cmin,rmax],[cmax,rmax]],dtype = "float32")
dst = np.array([[cmin-1024,rmin+128],[cmax-1024,rmin+128],[cmin+1024,rmax-64],[cmax+1024,rmax-64]],dtype = "float32")
M = cv2.getPerspectiveTransform(src, dst)
out_img = cv2.warpPerspective(result_img, M,(int(cols*1.5), int(rows*1.5)))
cv2.imwrite("project.jpg", out_img)


# rotate, and mirror image
result_img = cv2.imread('project.jpg')
result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
rows, cols, dim = result_img.shape
# uncomment the line below and comment the next line will rotate the ladder.
# angle = np.radians(30) 
angle = np.radians(0)
# transformation matrix for Rotation
M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
            	[np.sin(angle), np.cos(angle), 0],
            	[0, 0, 1]])

result = cv2.warpPerspective(result_img, M, (int(cols), int(rows)))

# uncomment the line below will mirror the ladder.
# result = cv2.flip(result, flipCode=-1)
cv2.imwrite("project_affine.jpg", result)


# find roi
img = cv2.imread("project_affine.jpg")
rmin, rmax, cmin, cmax = bbox(img)
roi = img[rmin: rmax+1, cmin: cmax+1]


# paste the ladder into the backgound
img = cv2.imread("back.jpg")
img_overlay_rgba = cv2.imread("project_affine.jpg")
img_overlay_rgba = cv2.cvtColor(img_overlay_rgba, cv2.COLOR_BGR2RGB)
img_overlay_rgba = cv2.resize(img_overlay_rgba, (360, 360))

# translation change x and y
# uncomment the line below and comment the next line will do the translation.
x, y = 1200, 1000 # default
# x, y = 1000, 1000
x_end = x + 360
y_end = y + 360


temp_img = img.copy()
small_img_gray = cv2.cvtColor(img_overlay_rgba, cv2.COLOR_RGB2GRAY)
small_img_gray = cv2.bitwise_not(small_img_gray)
ret, mask = cv2.threshold(small_img_gray, 120, 255, cv2.THRESH_BINARY)
bg = cv2.bitwise_or(temp_img[y:y_end,x:x_end], temp_img[y:y_end,x:x_end], mask = mask)
mask_inv = cv2.bitwise_not(small_img_gray)
fg = cv2.bitwise_and(img_overlay_rgba, img_overlay_rgba, mask=mask_inv)
final_roi = cv2.add(bg,fg)
img[y:y_end,x:x_end] = final_roi
cv2.imwrite("img_result.jpg", img)
