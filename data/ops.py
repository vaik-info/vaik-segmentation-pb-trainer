import numpy as np
import tensorflow as tf
import random
from PIL import Image, ImageOps


def random_resize(np_image, random_trials=2):
    org_np_image_size = np_image.shape
    for trial in range(random_trials):
        if random.uniform(0.0, 1.0) < 0.5:
            if min(np_image.shape[0], np_image.shape[1]) > 2:
                np_image = tf.image.resize(np_image, (int(np_image.shape[0] / 2), int(np_image.shape[1] / 2)))
    np_image = tf.image.resize(np_image, (org_np_image_size[0], org_np_image_size[1])).numpy()
    return np_image


def random_flip(raw_image, seg_image):
    np_image = np.concatenate([raw_image, seg_image], axis=-1)
    if random.uniform(0.0, 1.0) < 0.5:
        np_image = tf.image.flip_left_right(np_image).numpy()
    return np_image[:, :, :3], np_image[:, :, 3:]


def random_padding(raw_image, seg_image, random_ratio=(0.0, 0.05)):
    np_image = np.concatenate([raw_image, seg_image], axis=-1)
    left = int(np_image.shape[1] * random.uniform(random_ratio[0], random_ratio[1]))
    right = int(np_image.shape[1] * random.uniform(random_ratio[0], random_ratio[1]))
    top = int(np_image.shape[0] * random.uniform(random_ratio[0], random_ratio[1]))
    bottom = int(np_image.shape[0] * random.uniform(random_ratio[0], random_ratio[1]))
    np_image = tf.pad(np_image, tf.stack([[top, bottom], [left, right], [0, 0]])).numpy()
    return np_image[:, :, :3], np_image[:, :, 3:]


def random_hsv(image, max_delta=0.1, lower=2, upper=5, random_ratio=0.25):
    if random.uniform(0.0, 1.0) < random_ratio:
        image = tf.image.random_hue(image, max_delta)
        image = tf.image.random_saturation(image, lower, upper)
        image = image.numpy()
    return image


def data_valid(np_image):
    if 0 in np_image.shape:
        np_image = tf.zeros((1, 1, np_image.shape[-1]), dtype=np_image.dtype)
    return np_image


def resize_and_pad(np_image, image_size, resample=None):
    max_np_image = np.max(np_image)
    min_np_image = np.min(np_image)
    if np_image.shape[-1] == 1:
        pil_image = Image.fromarray(np.squeeze(np_image, axis=-1), 'L')
    else:
        pil_image = Image.fromarray(np_image)
    width, height = pil_image.size

    scale = image_size / max(width, height)

    resize_width = int(width * scale)
    resize_height = int(height * scale)

    pil_image = pil_image.resize((resize_width, resize_height), resample=resample)

    padding_width = image_size - resize_width
    padding_height = image_size - resize_height
    left = int(padding_width / 2)
    right = padding_width - left
    top = int(padding_height / 2)
    bottom = padding_height - top
    pil_image = ImageOps.expand(pil_image, (left, top, right, bottom), fill=0)
    np_image = np.asarray(pil_image)
    np_image = np.clip(np_image, min_np_image, max_np_image)
    return np_image


def resize_by_pool_size(np_image, pool_size):
    np_image = tf.image.resize(np_image, (int(np_image.shape[0] / pool_size), int(np_image.shape[1] / pool_size)), method=tf.image.ResizeMethod.BICUBIC).numpy()
    return np_image