import os
import argparse
import logging
import json
from datetime import datetime
import pytz
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_model_optimization.python.core.quantization.keras import quantize
import tensorflow_model_optimization as tfmot

tf.get_logger().setLevel('ERROR')
tf.debugging.disable_traceback_filtering()

from data import segmentation_dataset
from model import deeplab_v3_plus
from callbacks import save_callback, detail_logging_callback


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

model_dict = {
    'deeplab_v3_plus': deeplab_v3_plus.prepare
}

def load_weights(source_model, quant_aware_model):
    source_layers = source_model.layers
    quant_aware_layers = [layer for layer in quant_aware_model.layers if layer.name != 'quantize_layer']
    for layer_index in range(min(len(source_model.layers), len(quant_aware_model.layers))):
        source_layers_weight = source_layers[layer_index].get_weights()
        weight = quant_aware_layers[layer_index].get_weights()
        if len(source_layers_weight) == 0 or len(weight) == 0:
            continue
        weight[0] = source_layers_weight[0]
        quant_aware_layers[layer_index].set_weights(weight)

def train(load_weight_path, train_input_dir_path, valid_input_dir_path, classes_json_path, model_type, epochs, step_size, batch_size,
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
    model = model_dict[model_type](len(classes_dict['classes']), image_size, False)
    model.load_weights(load_weight_path)

    quant_aware_model = model_dict[model_type](len(classes_dict['classes']), image_size, True)

    with tfmot.quantization.keras.quantize_scope({'TpuConv2DQuantizeConfig': deeplab_v3_plus.TpuConv2DQuantizeConfig,
                                                  'TpuConv2DLayer': deeplab_v3_plus.TpuConv2DLayer,
                                                  }):
        quant_aware_model = tfmot.quantization.keras.quantize_apply(quant_aware_model)

    quant_aware_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tfa.losses.SigmoidFocalCrossEntropy(),
                      metrics=tf.keras.metrics.OneHotMeanIoU(len(classes_dict['classes']), ignore_class=classes_dict['classes'].index('background')))
    quant_aware_model.summary()

    load_weights(model, quant_aware_model)

    # prepare callback
    save_model_dir_path = os.path.join(output_dir_path,
                                       f'{datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d-%H-%M-%S")}')
    prefix = f'step-{step_size}_batch-{batch_size}'
    callbacks = [save_callback.SaveCallback(save_model_dir_path=save_model_dir_path, prefix=prefix),
                 detail_logging_callback.DetailLoggingCallback(classes_dict['classes'],
                                                               classes_dict['classes'].index('background'),
                                                               train_valid_data, valid_data, batch_size)]


    quant_aware_model.fit_generator(train_dataset, steps_per_epoch=step_size,
                        epochs=epochs,
                        validation_data=valid_data,
                        callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pb')
    parser.add_argument('--load_weight_path', type=str, default='~/output_model/2023-03-07-07-50-25/step-5000_batch-8_epoch-6_loss_0.0050_one_hot_mean_io_u_0.6972_val_loss_0.0064_val_one_hot_mean_io_u_0.6498/step-5000_batch-8_epoch-6_loss_0.0050_one_hot_mean_io_u_0.6972_val_loss_0.0064_val_one_hot_mean_io_u_0.6498')
    parser.add_argument('--train_input_dir_path', type=str, default='~/.vaik-mnist-segmentation-dataset/train')
    parser.add_argument('--valid_input_dir_path', type=str, default='~/.vaik-mnist-segmentation-dataset/valid')
    parser.add_argument('--classes_json_path', type=str, default='~/.vaik-mnist-segmentation-dataset/classes.json')
    parser.add_argument('--model_type', type=str, default='deeplab_v3_plus')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--step_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_max_sample', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--output_dir_path', type=str, default='~/output_model')
    args = parser.parse_args()

    args.load_weight_path = os.path.expanduser(args.load_weight_path)
    args.train_input_dir_path = os.path.expanduser(args.train_input_dir_path)
    args.valid_input_dir_path = os.path.expanduser(args.valid_input_dir_path)
    args.classes_json_path = os.path.expanduser(args.classes_json_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(args.output_dir_path, exist_ok=True)
    train(args.load_weight_path, args.train_input_dir_path, args.valid_input_dir_path, args.classes_json_path, args.model_type,
          args.epochs, args.step_size, args.batch_size, args.test_max_sample,
          args.image_size, args.output_dir_path)
