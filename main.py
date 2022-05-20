import numpy as np
import cv2
import matplotlib.pyplot as plt

# Подготовка с исходным изображением
original_image = cv2.imread("totoro.png")
img_0 = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Блюр (размытие изображения, решение проблемы раскрашивания по пикселям)
img = cv2.medianBlur(img_0, 5)
for i in range(8):
    img = cv2.medianBlur(img, 5)

# Кластеризация К-средних
vectorized = img.reshape((-1, 3))
vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 20  # = итоговое кол.-во цветов на картинке
attempts = 10
ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape(img.shape)  # Итоговое сегментированное изображение

edges = cv2.Canny(result_image, 1, 1)  # Получение границ, т.е. схемы рисунка (белые границы/чёрный фон)

invert = cv2.bitwise_not(edges)  # Получение чёрных границ на белом фоне

# Процесс добавления цифр
canvas = cv2.cvtColor(invert, cv2.COLOR_BGR2RGB) # Подготовка изображения

thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)  # thresh
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 2)  # Поиск контуров
n = 0  # Счётк для контуров

for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0 and hierarchy[0, n, 2] == -1: # Только контуры, у которых нет "детей"
        # Нахождение центров контуров
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if edges[cY][cX] == 0:  # Центр - не граница
            # Определение номера цвета
            i = 0
            while (result_image[cY][cX][0] != center[i][0]) and (result_image[cY][cX][1] != center[i][1]) and (result_image[cY][cX][2] != center[i][2]):
                i += 1
            # Добавление цифры(шрифт/размер/цвет)
            cv2.putText(canvas, str(i+1), (cX, cY), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (100, 149, 237), 1)
    n += 1

# Вывод результатов
figure_size = 15

plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(img_0)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(img)
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(img)
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(result_image)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray_r')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# Вывод цветов использованных в рисунке
for i in range(K):
    color = (center[i][0] / 255., center[i][1] / 255., center[i][2] / 255.)
    plt.scatter(i + 1, 1, color=color)
plt.show()

plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(result_image)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(thresh, cmap='gray_r')
plt.title('Thresh Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(result_image)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(canvas)
plt.title('Canvas Image'), plt.xticks([]), plt.yticks([])
plt.show()

# Сохранение результата (с цифрами)
canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
#isWritten = cv2.imwrite('result_image-panda-5.jpg', canvas)
