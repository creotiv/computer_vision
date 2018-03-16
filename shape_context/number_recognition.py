import numpy as np
import cv2
import sys
from scipy.spatial.distance import cdist, cosine
from shape_context import ShapeContext
import matplotlib.pyplot as plt

sc = ShapeContext()

def get_contour_bounding_rectangles(gray):
    """
      Getting all 2nd level bouding boxes based on contour detection algorithm.
    """
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for cnt in cnts[1]:
        (x, y, w, h) = cv2.boundingRect(cnt)
        res.append((x, y, x + w, y + h))

    return res

def parse_nums(sc, path):
    img = cv2.imread(path, 0)
    # invert image colors
    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    # making numbers fat for better contour detectiion
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    
    # getting our numbers one by one
    rois = get_contour_bounding_rectangles(img)
    grayd = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    nums = []
    for r in rois:
        grayd = cv2.rectangle(grayd, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 1)
        nums.append((r[0], r[1], r[2], r[3]))
    # we are getting contours in different order so we need to sort them by x1
    nums = sorted(nums, key=lambda x: x[0])
    descs = []
    for i, r in enumerate(nums):
        if img[r[1]:r[3], r[0]:r[2]].mean() < 50:
            continue
        points = sc.get_points_from_img(img[r[1]:r[3], r[0]:r[2]], 15)
        descriptor = sc.compute(points).flatten()
        descs.append(descriptor)
    return np.array(descs)

def match(base, current):
    """
      Here we are using cosine diff instead of "by paper" diff, cause it's faster
    """
    res = cdist(base, current.reshape((1, current.shape[0])), metric="cosine")
    char = str(np.argmin(res.reshape(11)))
    if char == '10':
        char = "/"
    return char

base_0123456789 = parse_nums(sc, '../resources/sc/numbers.png')
recognize = parse_nums(sc, '../resources/sc/numbers_test3.png')
res = ""
for r in recognize:
    res += match(base_0123456789, r)
img = cv2.imread('../resources/sc/numbers_test3.png')
plt.imshow(img)
plt.show()
print res
