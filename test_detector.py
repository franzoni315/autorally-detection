#!/usr/bin/python

import cv2, os
import numpy as np

class AutorallyDetector:
    def __init__(self):
        self.svm_model_name = os.path.join('svm.txt')
        svm_vectors = []
        with open('svm.txt') as f:
            for line in f:
                svm_vectors.append(float(line))

        #win_size = (128,64)
        win_size = (96,48)
        block_size = (16,16)
        block_stride = (8,8)
        cell_size = (8,8)
        nbins = 9
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        self.hog.setSVMDetector(np.asarray(svm_vectors))

    def detect(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, w = self.hog.detectMultiScale(gray_img, winStride=(16,16), padding=(0,0), scale=1.05)
        #if not w == ():
        #    max_ind = np.argmax(w)
        #    if w[max_ind] > 1.0:
        #        found = [found[np.argmax(w),:]]
        #    else:
        #        found =[]

        #boxes = []
        #if not w == ():
        #    for i, elem in enumerate(w):
        #        if elem >= 1:
        #            boxes.append(found[i,:])
        return found
        #return boxes

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness)

if __name__ == '__main__':
    detector = AutorallyDetector()
    capture = cv2.VideoCapture("/home/alexl/Documents/autorally-detection/AutorallyDetection3.mp4")
    # capture = cv2.VideoCapture("/home/igor/Downloads/test.mp4")

    cv2.namedWindow('video')
    while True:
        ret, im = capture.read()
        im = cv2.resize(im, (800, 600))
        found = detector.detect(im)
        draw_detections(im, found)
        cv2.imshow('video', im)
        if 0xFF & cv2.waitKey(5) == 27:
            break
