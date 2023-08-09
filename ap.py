"""
@Project ：mAP 
@File    ：ap.py 
@Author  ：46069
@Date    ：2023/8/8 11:34 
"""
import numpy as np


def voc_ap(rec: np.ndarray, prec: np.ndarray, use_07_metric: bool = False) -> float:
    """
    :param rec: 召回率  [N]
    :param prec: 准确率 [N]
    :param use_07_metric: 指标计算方式
    :return: ap值 float
    """
    if use_07_metric:
        # 07的计算方式：
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # 新的计算方式，计算PR曲线下面的面积
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
