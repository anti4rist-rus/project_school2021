"""Финальная функция преобразования изображения в текст"""

from typing import *
from open_png import *
from rotation import *


def img_to_str(model: Any, image_file: str):
    letters = open_letters(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i + 1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s_out += emnist_rotation_img(model, letters[i][2])
        if dn > letters[i][1] / 4:
            s_out += ' '
    return s_out
