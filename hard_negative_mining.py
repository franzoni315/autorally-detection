import cv2, os
import numpy as np
from test_detector import AutorallyDetector

class AutorallyHardNegativeMining:
    def __init__(self):
        self.hog_width = 128
        self.hog_height = 64
        self.database_path = 'autorally_database'
        self.counter = 2000

    def crop_image(self,img, rects, thickness = 1):
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

if __name__ == '__main__':
    detector = AutorallyDetector()
    negative_miner = AutorallyHardNegativeMining()
    capture = cv2.VideoCapture("/home/igor/py-faster-rcnn/data/demo/autorallyDetection2.mp4")
    cv2.namedWindow('video')
    while True:
        ret, im = capture.read()
        found = detector.detect(im)
        negative_miner.crop_image(im, found)
        # cv2.imshow('video', im)
        # if 0xFF & cv2.waitKey(5) == 27:
        #     break