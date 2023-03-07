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
## train_pb_aqt_tflite.py

### Usage

```shell
pip install -r requirements.txt
python train_pb_qat_tflite.py --load_weight_path ~/output_model/2023-03-06-08-33-31/step-5000_batch-8_epoch-9_loss_0.0046_one_hot_mean_io_u_0.7108_val_loss_0.0035_val_one_hot_mean_io_u_0.7478/step-5000_batch-8_epoch-9_loss_0.0046_one_hot_mean_io_u_0.7108_val_loss_0.0035_val_one_hot_mean_io_u_0.7478 \ 
                --train_input_dir_path ~/.vaik-mnist-segmentation-dataset/train \
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
## train_pb_aqt_trt.py

### Usage

```shell
pip install -r requirements.txt
./tensorflow_quantization_install.sh
python train_pb_qat_trt.py --load_weight_path ~/output_model/2023-03-06-08-33-31/step-5000_batch-8_epoch-9_loss_0.0046_one_hot_mean_io_u_0.7108_val_loss_0.0035_val_one_hot_mean_io_u_0.7478/step-5000_batch-8_epoch-9_loss_0.0046_one_hot_mean_io_u_0.7108_val_loss_0.0035_val_one_hot_mean_io_u_0.7478 \ 
                --train_input_dir_path ~/.vaik-mnist-segmentation-dataset/train \
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