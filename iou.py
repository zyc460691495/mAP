"""
@Project ：mAP
@File    ：iou.py
@Author  ：46069
@Date    ：2023/8/7 21:20
"""
import numpy as np


def compute_iou(true_boxes: np.ndarray, predict_box: np.ndarray) -> np.ndarray:
    """
    :param true_boxes: 真实框 [N,4]
    :param predict_box: 预测框 [4]
    :return: 返回某个预测框和所有真实框的交并比 [N]
    """
    assert len(true_boxes.shape) == 2 and true_boxes.shape[1] == 4
    assert predict_box.shape == (4,)

    # 使用广播机制，矩阵运算加速
    x_min = np.maximum(true_boxes[:, 0], predict_box[0])
    y_min = np.maximum(true_boxes[:, 1], predict_box[1])
    x_max = np.minimum(true_boxes[:, 2], predict_box[2])
    y_max = np.minimum(true_boxes[:, 3], predict_box[3])

    true_boxes_areas = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    predict_box_area = (predict_box[2] - predict_box[0]) * (predict_box[3] - predict_box[1])

    # 没有交集 x_max - x_min就会小于0
    intersection = np.maximum(0.0, x_max - x_min) * np.maximum(0.0, y_max - y_min)
    ious = intersection / (true_boxes_areas + predict_box_area - intersection)

    assert ious.shape == (true_boxes.shape[0],)
    return ious


if __name__ == '__main__':
    compute_iou(np.array([[0, 0, 50, 50], [50, 0, 100, 50]]), np.array([25, 0, 75, 50]))
