import numpy as np
import cv2
import matplotlib.pyplot as plt


def k_means(img, K, text):
    vectorized = img.reshape((-1, 3))
    vectorized = np.float32(vectorized)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape(img.shape)

    edges = cv2.Canny(result_image, 1, 1)
    return edges


original_image = cv2.imread("panda.jpg")
img_0 = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

K = 20
n = 8

kernel = np.ones((5, 5), np.uint8) / 25
img = cv2.filter2D(img_0, -1, kernel)
for i in range(n):
    img = cv2.filter2D(img, -1, kernel)
res_img1 = k_means(img, K, 'Segmented Image - filter2D')

img = cv2.bilateralFilter(img_0,9,75,75)
for i in range(n):
    img = cv2.bilateralFilter(img,9,75,75)
res_img2 = k_means(img, K, 'Segmented Image - bilateralFilter')

img = cv2.medianBlur(img_0, 5)
for i in range(n):
    img = cv2.medianBlur(img, 5)
res_img3 = k_means(img, K, 'Segmented Image - medianBlur')

img = cv2.GaussianBlur(img_0,(5,5),0)
for i in range(n):
    img = cv2.GaussianBlur(img,(5,5),0)
res_img4 = k_means(img, K, 'Segmented Image - GaussianBlur')

figure_size = 15

plt.figure(figsize=(figure_size, figure_size))
plt.subplot(2, 3, 5), plt.imshow(res_img1, cmap='gray_r')
plt.title('Segmented Image - filter2D'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(res_img2, cmap='gray_r')
plt.title('Segmented Image - bilateralFilter'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(res_img3, cmap='gray_r')
plt.title('Segmented Image - medianBlur'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(res_img4, cmap='gray_r')
plt.title('Segmented Image - GaussianBlur'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 1), plt.imshow(img_0, cmap='gray_r')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.show()
