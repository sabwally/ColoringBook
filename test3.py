import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("a2.png")
img_0 = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

img = cv2.medianBlur(img_0, 5)
for i in range(8):
    img = cv2.medianBlur(img, 5)
vectorized = img.reshape((-1, 3))
vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 15
attempts = 10
ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape(img.shape)

edges = cv2.Canny(result_image, 1, 1)

for n in range(K):
    color = center[n]
    img_1 = result_image.copy()
    img_canvas = img_1.copy()
    p = 255
    t = 250
    if color[0] == 255 and color[1] == 255 and color[2] == 255:
        p = 0
    i_y = 0
    for y in img_1:
        i_x = 0
        for x in y:
            if (x[0] != color[0]) or (x[1] != color[1]) or (x[2] != color[2]):
                img_canvas[i_y][i_x][0] = p
                img_canvas[i_y][i_x][1] = p
                img_canvas[i_y][i_x][2] = p
            i_x += 1
        i_y += 1

    gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    (T, thresh) = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if edges[cY][cX] == 0 and (result_image[cY][cX][0] == color[0] and result_image[cY][cX][1] == color[1] and result_image[cY][cX][2] == color[2]):
                cv2.putText(edges, str(n+1), (cX, cY), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)
            # print(cX, cY)
    '''
    figure_size = 15
    plt.figure(figsize=(figure_size, figure_size))
    plt.subplot(1, 2, 1), plt.imshow(gray)
    plt.title('Gray'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(thresh, cmap='gray_r')
    plt.title('Threshold %i' % (n+1)), plt.xticks([]), plt.yticks([])
    plt.show()
    '''

'''
for i in range(K):
    color = (center[i][0] / 255., center[i][1] / 255., center[i][2] / 255.)
    plt.scatter(i+1, 1, color=color)
'''
figure_size = 15
plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(result_image)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray_r')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()