"""
@Project ：mAP 
@File    ：mAP.py 
@Author  ：46069
@Date    ：2023/8/9 10:11 
"""
import os
import pickle

import numpy as np
from ap import voc_ap
from iou import compute_iou


def voc_eval(detpath, annopath, classname, cachedir, ovthresh=0.5, use_07_metric=False):
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)

    cachefile = os.path.join(cachedir, 'annots.pkl')

    imagenames = [filename[:-4] for filename in os.listdir("./data/images-optional")]

    if not os.path.isfile(cachefile):

        recs = {}
        dets = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = [line.split() for line in open(annopath + "{}.txt".format(imagename)).readlines()]
            dets[imagename] = [line.split() for line in open(detpath + "{}.txt".format(imagename)).readlines()]
            if i % 10 == 0:
                print('Reading for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump((recs, dets), f)
    else:
        with open(cachefile, 'rb') as f:
            try:
                recs, dets = pickle.load(f)
            except:
                recs, dets = pickle.load(f, encoding='bytes')

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj[0] == classname]
        bbox = np.array([[int(y) for y in x[1:]] for x in R])
        det = [False] * len(R)
        class_recs[imagename] = {'bbox': bbox, 'det': det}
        npos += len(bbox)

    lines = []
    for imagename in imagenames:
        for obj in dets[imagename]:
            if obj[0] == classname:
                lines.append([imagename, *obj])

    image_ids = [x[0] for x in lines]
    confidence = np.array([float(x[2]) for x in lines])
    BB = np.array([[float(z) for z in x[3:]] for x in lines])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ious = compute_iou(BBGT, bb)
                ovmax = np.max(ious)
                jmax = np.argmax(ious)

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


if __name__ == '__main__':
    print(voc_eval("./data/detection-results/", "./data/ground-truth/", "11", "./cache"))
