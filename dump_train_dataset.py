import os
import argparse
import json
from tqdm import tqdm
from PIL import Image
import numpy as np

from data import segmentation_dataset


def dump(input_image_dir_path, classes_json_path, sample_num, image_size, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    with open(classes_json_path, 'r') as f:
        classes_dict = json.load(f)

    raw_output_image_dir_path = os.path.join(output_dir_path, 'raw')
    os.makedirs(raw_output_image_dir_path, exist_ok=True)
    seg_output_image_dir_path = os.path.join(output_dir_path, 'seg')
    os.makedirs(seg_output_image_dir_path, exist_ok=True)

    TrainDataset = type(f'TrainDataset', (segmentation_dataset.SegmentationDataset,), dict())
    train_dataset = TrainDataset(input_image_dir_path, classes_dict['classes'], classes_dict['colors'], image_size)
    train_dataset = iter(train_dataset)
    for index in tqdm(range(sample_num)):
        image, seg_image = train_dataset.get_next()
        image = image.numpy()
        output_raw_image_path = os.path.join(raw_output_image_dir_path, f'{index:09d}.png')
        Image.fromarray(image).save(output_raw_image_path, quality=100, subsampling=0)
        seg_image = np.squeeze(seg_image.numpy(), axis=-1)
        output_seg_image_path = os.path.join(seg_output_image_dir_path, f'{index:09d}_seg.png')
        Image.fromarray(seg_image).save(output_seg_image_path, quality=100, subsampling=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--input_image_dir_path', type=str, default='~/.vaik-mnist-segmentation-dataset/train')
    parser.add_argument('--classes_json_path', type=str, default='~/.vaik-mnist-segmentation-dataset/classes.json')
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-segmentation-pb-trainer/dump_train')
    args = parser.parse_args()

    args.input_image_dir_path = os.path.expanduser(args.input_image_dir_path)
    args.classes_json_path = os.path.expanduser(args.classes_json_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    dump(args.input_image_dir_path, args.classes_json_path, args.sample_num, args.image_size, args.output_dir_path)
