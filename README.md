# vaik-segmentation-pb-trainer

Train segmentation pb model

## train_pb.py

### Usage

```shell
pip install -r requirements.txt
python train_pb.py --train_input_dir_path ~/.vaik-mnist-segmentation-dataset/train \
                --valid_input_dir_path ~/.vaik-mnist-segmentation-dataset/valid \
                --classes_json_path ~/.vaik-mnist-segmentation-dataset/classes.json \
                --model_type deeplab_v3_plus \
                --epochs 100 \
                --step_size 5000 \
                --batch_size 8 \
                --test_max_sample 100 \
                --image_size 320 \
                --output_dir_path '~/output_model'        
```

- train_input_dir_path & valid_input_dir_path

```shell
train/
├── train_000000000_raw.png
├── train_000000000_seg.png
├── train_000000001_raw.png
├── train_000000001_seg.png
├── train_000000002_raw.png
・・・
```

### Output

![vaik-segmentation-pb-trainer-output-train1](https://user-images.githubusercontent.com/116471878/200271108-3b485be9-be4d-48f3-b185-855be8651cf6.png)

![vaik-segmentation-pb-trainer-output-train2](https://user-images.githubusercontent.com/116471878/200271111-f21fc130-02f1-4d6d-b609-26884ebb9c59.png)
 
-----

## dump_train_dataset.py

### Usage

```shell
pip install -r requirements.txt
python dump_train_dataset.py --input_image_dir_path ~/.vaik-mnist-segmentation-dataset/valid \
                --classes_json_path ~/.vaik-mnist-segmentation-dataset/classes.json \
                --sample_num 1000 \
                --image_size 320 \
                --output_dir_path ~/.vaik-segmentation-pb-trainer/dump_train
```
### Output

![Screenshot from 2023-03-04 22-39-50](https://user-images.githubusercontent.com/116471878/222905607-2649cb13-72b7-4819-a99d-b518a7e77a84.png)

![Screenshot from 2023-03-04 22-39-16](https://user-images.githubusercontent.com/116471878/222905601-67d24375-36cd-4812-89d5-22be32918c46.png)

![Screenshot from 2023-03-04 22-39-27](https://user-images.githubusercontent.com/116471878/222905604-ad857f17-2315-4fc3-80d1-be6b175f69ef.png)

![Screenshot from 2023-03-04 22-39-40](https://user-images.githubusercontent.com/116471878/222905606-90301915-a4ee-4a35-9c3d-869951dbd942.png)
