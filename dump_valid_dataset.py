import os
import argparse
import json
import colorsys
from tqdm import tqdm
from PIL import Image
import numpy as np

from data import segmentation_dataset


def get_classes_color(classes):
    colors = [[0, 0, 0]]
    for classes_index in range(1, len(classes)):
        rgb = colorsys.hsv_to_rgb(classes_index / len(classes), 1, 1)
        colors.append([int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)])
    return colors


def dump(input_image_dir_path, classes_json_path, test_max_sample, image_size, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    with open(classes_json_path, 'r') as f:
        classes_dict = json.load(f)

    raw_output_image_dir_path = os.path.join(output_dir_path, 'raw')
    os.makedirs(raw_output_image_dir_path, exist_ok=True)
    seg_output_image_dir_path = os.path.join(output_dir_path, 'seg')
    os.makedirs(seg_output_image_dir_path, exist_ok=True)
    vis_output_image_dir_path = os.path.join(output_dir_path, 'vis')
    os.makedirs(vis_output_image_dir_path, exist_ok=True)

    colors = get_classes_color(classes_dict['classes'])
    json_dict = {'classes': classes_dict['classes'], 'colors': colors}
    with open(os.path.join(vis_output_image_dir_path, 'classes.json'), 'w') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    TestDataset = type(f'TestDataset', (segmentation_dataset.TestSegmentationDataset,), dict())
    valid_dataset = TestDataset(input_image_dir_path, classes_dict['classes'], classes_dict['colors'], image_size,
                                max_sample=test_max_sample)
    valid_data = segmentation_dataset.TestSegmentationDataset.get_all_data(valid_dataset)
    for index in tqdm(range(valid_data[0].shape[0])):
        image, seg_image = valid_data[0][index], valid_data[1][index]
        output_raw_image_path = os.path.join(raw_output_image_dir_path, f'{index:09d}.png')
        Image.fromarray(image).save(output_raw_image_path, quality=100, subsampling=0)

        seg_image = np.argmax(seg_image, axis=-1).astype(np.uint8)
        output_seg_image_path = os.path.join(seg_output_image_dir_path, f'{index:09d}_seg.png')
        Image.fromarray(seg_image, 'L').save(output_seg_image_path, quality=100, subsampling=0)

        vis_image = np.zeros(seg_image.shape + (3,), dtype=np.uint8)
        for class_index in range(len(classes_dict['classes'])):
            array_index = seg_image == class_index
            vis_image[array_index] = colors[class_index]
        output_vis_image_path = os.path.join(vis_output_image_dir_path, f'{index:09d}_vis.png')
        Image.fromarray(vis_image).save(output_vis_image_path, quality=100, subsampling=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--input_image_dir_path', type=str, default='~/.vaik-mnist-segmentation-dataset/valid')
    parser.add_argument('--classes_json_path', type=str, default='~/.vaik-mnist-segmentation-dataset/classes.json')
    parser.add_argument('--test_max_sample', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-segmentation-pb-trainer/dump_valid')
    args = parser.parse_args()

    args.input_image_dir_path = os.path.expanduser(args.input_image_dir_path)
    args.classes_json_path = os.path.expanduser(args.classes_json_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    dump(args.input_image_dir_path, args.classes_json_path, args.test_max_sample, args.image_size, args.output_dir_path)
