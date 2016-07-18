#!/usr/bin/python

import cv2, os, sys
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

class AutorallyDetectorMultiClass:
    def __init__(self):
        self.svm_ = joblib.load('svm.pkl')
        self.Kpos = 6
        self.Kneg = 1
        self.win_size = (96,48)
        block_size = (16,16)
        block_stride = (8,8)
        cell_size = (8,8)
        nbins = 9
        self.hog = cv2.HOGDescriptor(self.win_size, block_size, block_stride, cell_size, nbins)

    def detectMultiScale(self, img):
        found = []
        weights = []
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scales = np.linspace(1,6,10)
        for scale in scales:
            found_, weights_ = self.detect(gray_img, scale)
            found += found_
            weights += weights_
        print weights
        return found, weights

    def detect(self, img, scale=1, win_stride=(32,16)):
        found = []
        weights = []
        rows = int(img.shape[0]/scale)
        cols = int(img.shape[1]/scale)
        win_stride = (int(np.floor(win_stride[0]/np.sqrt(scale))), int(np.floor(win_stride[1]/np.sqrt(scale))))
        img = cv2.resize(img, (cols,rows))
        width_blocks = 0
        height_blocks = 0
        width = 96
        height = 48
        while width <= cols:
            width += win_stride[0]
            width_blocks += 1
        while height <= rows:
            height += win_stride[1]
            height_blocks += 1
        features = self.hog.compute(img, win_stride)
        features = features.reshape((features.size/self.hog.getDescriptorSize(), self.hog.getDescriptorSize()))
        predictions = self.svm_.decision_function(features)
        for i in range(height_blocks * width_blocks):
            col = i % width_blocks
            row = i / width_blocks
            prediction = predictions[i]
            neg_weight = 0
            for j in range(self.Kpos, self.Kpos + self.Kneg):
                if prediction[j] > 0:
                    neg_weight += prediction[j]
            if neg_weight > 0:
                continue
            pos_weight = 0
            for j in range(self.Kpos):
                if prediction[j] > 0:
                    pos_weight += prediction[j]
            if pos_weight > 0.2:
                x = int(col*win_stride[0]*scale)
                y = int(row*win_stride[1]*scale)
                w = int(self.win_size[0]*scale)
                h = int(self.win_size[1]*scale)
                found.append([x, y, w, h])
                weights.append(pos_weight)

        return found, weights

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness)

def non_max_suppression_fast(boxes, weights, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    w = boxes[:,2]
    h = boxes[:,3]
    x2 = x1 + w
    y2 = y1 + h

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

if __name__ == '__main__':
    detector = AutorallyDetectorMultiClass()
    capture = cv2.VideoCapture("/home/igor/Documents/autorally-detection/autorally_database/Videos/0002.mp4")
    # capture = cv2.VideoCapture("/home/igor/Downloads/test.mp4")

    cv2.namedWindow('video')
    while True:
        ret, im = capture.read()

        im = cv2.resize(im, (96*8, 48*12))
        found, w = detector.detectMultiScale(im)
        boxes = non_max_suppression_fast(np.asarray(found), np.asarray(w), 0.3)
        draw_detections(im, boxes)
        cv2.imshow('video', im)
        if 0xFF & cv2.waitKey(1) == 27:
            break