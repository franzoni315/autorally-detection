#!/usr/bin/python

import cv2, os
import numpy as np
from test_detector_subcategories import AutorallyDetectorMultiClass, non_max_suppression_fast

class AutorallyHardNegativeMining:
    def __init__(self):
        self.hog_width = 128
        self.hog_height = 64
        self.database_path = 'autorally_database'
        self.counter = 5000

    def crop_image(self,img, rects, thickness = 1):
        print rects
        for x, y, w, h in rects:
            save_name = os.path.join(self.database_path, 'HardNegativeMiningNew', '%05d' % self.counter + 'hdn.jpg')
            self.counter += 1
            xstart = x
            ystart = y
            xend = x + w
            yend = y + h
            rows, cols, channels = img.shape
            if xstart < 0:
                xstart = 0
            if ystart < 0:
                ystart = 0
            if xend > cols:
                xend = cols - 1
            if yend > rows:
                yend = rows - 1
            crop_img = img[ystart:yend, xstart:xend]
            res_img = cv2.resize(crop_img,(self.hog_width, self.hog_height))
            cv2.imwrite(save_name, res_img)

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness)

if __name__ == '__main__':
    detector = AutorallyDetectorMultiClass()
    negative_miner = AutorallyHardNegativeMining()
    capture = cv2.VideoCapture("/home/igor/Documents/autorally-detection/autorally_database/Videos/left_camera_image_color_compressed2.mp4")
    cv2.namedWindow('video')
    while True:
        ret, im = capture.read()

        im = cv2.resize(im, (96*8, 48*12))
        found, w = detector.detectMultiScale(im)
        boxes = non_max_suppression_fast(np.asarray(found), np.asarray(w), 0.3)
        negative_miner.crop_image(im, boxes)
        draw_detections(im, boxes, 3)
        cv2.imshow('video', im)
        if 0xFF & cv2.waitKey(5) == 27:
            break
