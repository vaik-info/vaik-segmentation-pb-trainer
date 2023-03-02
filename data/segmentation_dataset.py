import tensorflow as tf
import glob
import os
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
from data import ops


class SegmentationDataset:
    classes = None
    colors = None
    image_dict = None
    output_signature = None
    min_size = None
    image_size = None
    aug_ratio = 0.5

    def __new__(cls, input_dir_path, classes, colors, image_size=None, min_size=32):
        cls.classes = classes
        cls.colors = colors
        cls.image_size = image_size
        cls.image_dict = cls._prepare_image_dict(input_dir_path, classes, colors)
        cls.min_size = min_size
        cls.output_signature = (
            tf.TensorSpec(name=f'raw_image', shape=(image_size, image_size, 3), dtype=tf.uint8),
            tf.TensorSpec(name=f'output_image', shape=(image_size, image_size, 1), dtype=tf.int32)
        )
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=cls.output_signature
        )
        return dataset

    @classmethod
    def _generator(cls):
        while True:
            class_index = random.choice(list(range(len(cls.classes))))
            raw_image_path, seg_image_path = random.choice(cls.image_dict[cls.classes[class_index]])
            np_raw_image = tf.image.decode_image(tf.io.read_file(raw_image_path), channels=3).numpy()
            np_seg_image = tf.image.decode_image(tf.io.read_file(seg_image_path), channels=1).numpy()
            if random.uniform(0.0, 1.0) < cls.aug_ratio:
                np_raw_image, np_seg_image = cls._data_aug(np_raw_image, np_seg_image)
            np_raw_image = ops.resize_and_pad(np_raw_image, cls.image_size)
            np_seg_image = ops.resize_and_pad(np_seg_image, cls.image_size, resample=Image.NEAREST)
            if np_raw_image.shape[0] < cls.min_size or np_raw_image.shape[1] < cls.min_size:
                print(f'Pass {raw_image_path} because too small')
                continue
            yield (
                tf.convert_to_tensor(np_raw_image.astype(np.uint8)),
                tf.convert_to_tensor(np_seg_image.astype(np.int32))
            )

    @classmethod
    def _prepare_image_dict(cls, input_dir_path, classes, colors):
        image_dict = {}
        for class_label in classes:
            image_dict[class_label] = []
        raw_image_path_list = glob.glob(os.path.join(input_dir_path, f'**/*_raw.*'), recursive=True)

        for raw_image_path in tqdm(raw_image_path_list, desc='_prepare_image_dict()'):
            seg_image_path = raw_image_path.replace('raw', 'seg')
            seg_image = tf.image.decode_image(tf.io.read_file(seg_image_path), channels=1).numpy()
            seg_image_indexes = np.unique(seg_image)
            for seg_image_index in seg_image_indexes:
                image_dict[classes[seg_image_index]].append([raw_image_path, seg_image_path])
        return image_dict

    @classmethod
    def _data_aug(cls, raw_image: np.array, seg_image: np.array, random_r_ratio=0.25):
        raw_image = ops.random_resize(raw_image)
        # raw_image, seg_image = ops.random_flip(raw_image, seg_image)
        raw_image, seg_image = ops.random_padding(raw_image, seg_image)
        raw_image = ops.random_hsv(raw_image, random_ratio=random_r_ratio)
        return raw_image.astype(np.uint8), seg_image.astype(np.uint8)


class TestSegmentationDataset(SegmentationDataset):
    max_sample = None

    def __new__(cls, input_dir_path, classes, colors, image_size=None, min_size=32, max_sample=100):
        cls.max_sample = max_sample
        return super(TestSegmentationDataset, cls).__new__(cls, input_dir_path, classes, colors, image_size, min_size)

    @classmethod
    def _generator(cls):
        for class_index, class_label in enumerate(cls.classes):
            for image_index, (raw_image_path, seg_image_path) in enumerate(cls.image_dict[class_label]):
                if image_index > cls.max_sample - 1:
                    break
                np_raw_image = tf.image.decode_image(tf.io.read_file(raw_image_path), channels=3).numpy()
                np_seg_image = tf.image.decode_image(tf.io.read_file(seg_image_path), channels=1).numpy()
                if random.uniform(0.0, 1.0) < cls.aug_ratio:
                    np_raw_image, np_seg_image = cls._data_aug(np_raw_image, np_seg_image)
                np_raw_image = ops.resize_and_pad(np_raw_image, cls.image_size)
                np_seg_image = ops.resize_and_pad(np_seg_image, cls.image_size, resample=Image.NEAREST)

                if np_raw_image.shape[0] < cls.min_size or np_raw_image.shape[1] < cls.min_size:
                    print(f'Pass {raw_image_path} because too small')
                    continue
                yield (
                    tf.convert_to_tensor(np_raw_image.astype(np.uint8)),
                    tf.convert_to_tensor(np_seg_image.astype(np.int32))
                )

    @classmethod
    def _prepare_image_dict(cls, input_dir_path, classes, colors):
        image_dict = {}
        for class_label in classes:
            image_dict[class_label] = []
        raw_image_path_list = glob.glob(os.path.join(input_dir_path, f'**/*_raw.*'), recursive=True)
        random.shuffle(raw_image_path_list)
        for raw_image_path in raw_image_path_list:
            seg_image_path = raw_image_path.replace('raw', 'seg')
            image_dict[classes[0]].append([raw_image_path, seg_image_path])
        return image_dict

    @classmethod
    def get_all_data(cls, dataset):
        dataset = iter(dataset)
        raw_image_list, seg_image_list = [], []
        for data in dataset:
            raw_image_list.append(data[0])
            seg_image_list.append(data[1])
        return np.stack(raw_image_list), np.stack(seg_image_list)
