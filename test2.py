import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("mini_panda.jpg")
img_0 = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

img = cv2.medianBlur(img_0, 5)
for i in range(8):
    img = cv2.medianBlur(img, 5)

vectorized = img.reshape((-1, 3))
vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
attempts = 10
ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape(img.shape)

edges = cv2.Canny(result_image, 1, 1)
invert = cv2.bitwise_not(edges)
canvas = cv2.cvtColor(invert, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)

thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
#(T, thresh) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 2)
n = 0

for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0 and hierarchy[0, n, 2] == -1:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if edges[cY][cX] == 0:
            i = 0
            while (result_image[cY][cX][0] != center[i][0]) and (result_image[cY][cX][1] != center[i][1]) and (result_image[cY][cX][2] != center[i][2]):
                i += 1
            cv2.putText(canvas, str(i+1), (cX, cY), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (100, 149, 237), 1) #добавление цифр(шрифт/размер/цвет)
        #print(cX, cY)
    n += 1

figure_size = 15
for i in range(K):
    color = (center[i][0] / 255., center[i][1] / 255., center[i][2] / 255.)
    plt.scatter(i+1, 1, color=color)

plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(result_image)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(canvas)
plt.title('Canvas Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(gray)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(thresh, cmap='gray_r')
plt.title('Thresh Image'), plt.xticks([]), plt.yticks([])
plt.show()

canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
#isWritten = cv2.imwrite('result_image-panda-5.jpg', canvas)
