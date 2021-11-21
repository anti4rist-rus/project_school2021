"""main"""

from img_to_str import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


model = keras.models.load_model('DB machine/emnist_letters.h5')
s_out = img_to_str(model, "pic/text3.png")
print(s_out)


