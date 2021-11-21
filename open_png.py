"""Загрузка и преобразование изображения"""

import cv2
import numpy as np
from keras import *


def open_letters(image_file: str, out_size=28):
    img = cv2.imread(image_file)     # Открыть фото
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Преобразовать в ч/б (COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # Функция threshold сделало наше изображение жирнее, для лучшего распознавания в дальнейшем
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    # Увеличим текст
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)     # Сделаем контур

    output = img.copy()

    letters = []
    # Отображает контур букв на картинке
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)
            # преобразовывает в квадраты буквы
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Увеличим изображение сверху-снизу
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Увеличить изображение свлева-вправо
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop
                # Изменим размер буквы до 28x28 и добавим букву и ее координату X
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

            # Сортируем буквы по X
            letters.sort(key=lambda x: x[0], reverse=False)

    # Результатом является массив букв и их индексов
    """cv2.imshow("Input", img)
    cv2.imshow("Gray", thresh)
    cv2.imshow("Enlarged", img_erode)
    cv2.imshow("Output", output)
    cv2.imshow("0", letters[0][2])
    cv2.imshow("1", letters[1][2])
    cv2.imshow("2", letters[2][2])
    cv2.imshow("3", letters[3][2])
    cv2.imshow("4", letters[4][2])
    cv2.waitKey(0)"""

    return letters


"""open_letters('text1.png')"""
