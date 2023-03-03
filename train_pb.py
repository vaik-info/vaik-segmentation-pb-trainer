import os
import argparse
import logging
import json
from datetime import datetime
import pytz
import tensorflow as tf
import tensorflow_addons as tfa

tf.get_logger().setLevel('ERROR')
tf.debugging.disable_traceback_filtering()

from data import segmentation_dataset
from model import deeplab_v3_plus
from callbacks import save_callback, detail_logging_callback

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

model_dict = {
    'deeplab_v3_plus': deeplab_v3_plus.prepare
}

def train(train_input_dir_path, valid_input_dir_path, classes_json_path, model_type, epochs, step_size, batch_size,
          test_max_sample, image_size, output_dir_path):
    with open(classes_json_path, 'r') as f:
        classes_dict = json.load(f)
    # train
    TrainDataset = type(f'TrainDataset', (segmentation_dataset.SegmentationDataset,), dict())
    train_dataset = TrainDataset(train_input_dir_path, classes_dict['classes'], classes_dict['colors'], image_size)
    train_valid_data = next(iter(train_dataset.padded_batch(batch_size=test_max_sample, padding_values=(tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.float32)))))
    train_dataset = train_dataset.padded_batch(batch_size=batch_size, padding_values=(tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.float32)))

    # valid
    TestDataset = type(f'TestDataset', (segmentation_dataset.TestSegmentationDataset,), dict())
    valid_dataset = TestDataset(valid_input_dir_path, classes_dict['classes'], classes_dict['colors'], image_size,
                                max_sample=test_max_sample)
    valid_data = segmentation_dataset.TestSegmentationDataset.get_all_data(valid_dataset)

    # prepare model
    model = model_dict[model_type](len(classes_dict['classes']), image_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tfa.losses.SigmoidFocalCrossEntropy(),
                  metrics=tf.keras.metrics.OneHotMeanIoU(len(classes_dict['classes']), ignore_class=classes_dict['classes'].index('background')))
    # prepare callback
    save_model_dir_path = os.path.join(output_dir_path,
                                       f'{datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d-%H-%M-%S")}')
    prefix = f'step-{step_size}_batch-{batch_size}'
    callbacks = [save_callback.SaveCallback(save_model_dir_path=save_model_dir_path, prefix=prefix),
                 detail_logging_callback.DetailLoggingCallback(classes_dict['classes'],
                                                               classes_dict['classes'].index('background'),
                                                               train_valid_data, valid_data, batch_size)]


    model.fit_generator(train_dataset, steps_per_epoch=step_size,
                        epochs=epochs,
                        validation_data=valid_data,
                        callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pb')
    parser.add_argument('--train_input_dir_path', type=str, default='~/.vaik-mnist-segmentation-dataset/train')
    parser.add_argument('--valid_input_dir_path', type=str, default='~/.vaik-mnist-segmentation-dataset/valid')
    parser.add_argument('--classes_json_path', type=str, default='~/.vaik-mnist-segmentation-dataset/classes.json')
    parser.add_argument('--model_type', type=str, default='deeplab_v3_plus')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--step_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_max_sample', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--output_dir_path', type=str, default='~/output_model')
    args = parser.parse_args()

    args.train_input_dir_path = os.path.expanduser(args.train_input_dir_path)
    args.valid_input_dir_path = os.path.expanduser(args.valid_input_dir_path)
    args.classes_json_path = os.path.expanduser(args.classes_json_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(args.output_dir_path, exist_ok=True)
    train(args.train_input_dir_path, args.valid_input_dir_path, args.classes_json_path, args.model_type,
          args.epochs, args.step_size, args.batch_size, args.test_max_sample,
          args.image_size, args.output_dir_path)
