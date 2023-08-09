"""
@Project ：mAP 
@File    ：img_show.py 
@Author  ：46069
@Date    ：2023/8/8 10:18 
"""

import cv2
import os

import numpy as np


def plot_img(filename) -> None:
    """
    :param filename: 文件名不包含后缀
    :return: None
    """
    img_path = os.path.join("./data/images-optional", filename + ".jpg")
    anno_path = os.path.join("./data/ground-truth", filename + ".txt")
    det_path = os.path.join("./data/detection-results", filename + ".txt")

    annos = [line.split() for line in open(anno_path).readlines()]
    dets = [line.split() for line in open(det_path).readlines()]

    img = cv2.imread(img_path)

    for anno in annos:
        category = anno[0]
        x_min = int(anno[1])
        y_min = int(anno[2])
        x_max = int(anno[3])
        y_max = int(anno[4])

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, category, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    for det in dets:
        category = det[0]
        confidence = float(det[1])
        x_min = int(det[2])
        y_min = int(det[3])
        x_max = int(det[4])
        y_max = int(det[5])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(img, "{} {:.2f}".format(category, confidence), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 1)
    cv2.imshow("img", img)
    cv2.waitKey()


if __name__ == '__main__':
    plot_img("2007_000129")
