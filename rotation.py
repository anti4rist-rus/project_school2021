"""Поворот букв, чтобы соответствовать базе данных"""

import numpy as np
from model import *

model = keras.models.load_model('DB machine/emnist_letters_err.h5')


def emnist_rotation_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    result = model.predict_classes([img_arr])
    return chr(emnist_labels[result[0]])
