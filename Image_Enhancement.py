import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("t.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

HistEq = cv2.equalizeHist(gray)
binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] # binarize the image

kernel = np.ones((3, 3), np.uint8) # define the kernel

opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1) # opening the image
cv2.imwrite("GrayImg.jpg", gray)
cv2.imwrite("HistogramEqualization.jpg", HistEq)
cv2.imwrite("MorphologicalOperation.jpg", opening)
True

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))  # Adjust the figure size if needed

plt.subplot(2, 2, 1)
org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(org)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title("Gray Image")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(HistEq, cmap='gray')
plt.title("Histogram Equalization")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(opening, cmap='gray')
plt.title("Morphological Operation")
plt.axis('off')

plt.tight_layout()  # Adjust layout spacing
plt.show()
