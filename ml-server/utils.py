import os

# os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import load_model
import keras.backend as K

import numpy as np
from tensorflow.python.keras.losses import binary_crossentropy


print(K.backend())


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def get_model(model_path='model.h5'):
    model = load_model(model_path, compile=False)

    model.compile(optimizer='nadam', loss=bce_dice_loss, metrics=[dice_coef])

    return model


def make_predict(img, model, label):
    pad_width=((32, 32), (112, 112), (0, 0))
    pad_img = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)

    pad_img_mask = model.predict(pad_img[:, :, :3].reshape(1, 320, 480, 3))

    img_mask = pad_img_mask[0, 32: -32, 112:-112]
    id_mask = img_mask.argmax(axis=2)

    return (id_mask == label).astype(int)
