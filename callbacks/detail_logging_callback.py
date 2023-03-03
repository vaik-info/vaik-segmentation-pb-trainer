import numpy as np
import tensorflow as tf


class DetailLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, classes, ignore_index, train_valid_data, valid_data, batch_size):
        super(DetailLoggingCallback, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.train_valid_data = train_valid_data
        self.valid_data = valid_data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        pred = np.argmax(self.model.predict(self.train_valid_data[0], batch_size=self.batch_size), -1)
        gt = np.argmax(self.train_valid_data[1].numpy(), -1)
        train_result = self.__mean_iou(pred, gt, len(self.classes), self.ignore_index)

        pred = np.argmax(self.model.predict(self.valid_data[0], batch_size=self.batch_size), -1)
        gt = np.argmax(self.valid_data[1], -1)
        valid_result = self.__mean_iou(pred, gt, len(self.classes),  self.ignore_index)

        print(f'train_mIoU: {train_result[0]}, valid_mIoU: {valid_result[0]}')
        print('train_detail_IoU')
        self.__dump(self.classes, train_result[2], valid_result[2])

    def __dump(self, classes, train_iou_array, valid_iou_array):
        for index, label in enumerate(classes):
            print(f'{label}: train({train_iou_array[index]:.4f}), valid({valid_iou_array[index]:.4f})')

    def __intersect_and_union(self, pred_label, label, num_classes, ignore_index):
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
        area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
        area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
        area_union = area_pred_label + area_label - area_intersect

        return area_intersect, area_union, area_pred_label, area_label

    def __mean_iou(self, results, gt_seg_maps, num_classes, ignore_index=0):
        num_imgs = results.shape[0]
        total_area_intersect = np.zeros((num_classes,), dtype=np.float32)
        total_area_union = np.zeros((num_classes,), dtype=np.float32)
        total_area_pred_label = np.zeros((num_classes,), dtype=np.float32)
        total_area_label = np.zeros((num_classes,), dtype=np.float32)
        for i in range(num_imgs):
            area_intersect, area_union, area_pred_label, area_label = \
                self.__intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                           ignore_index=-1)
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label
        acc = total_area_intersect / total_area_label
        iou = total_area_intersect / total_area_union

        mask = np.ones(iou.shape, dtype=bool)
        mask[ignore_index] = False
        return np.mean(iou[mask]), acc, iou
