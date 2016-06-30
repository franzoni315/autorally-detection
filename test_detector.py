import cv2, os
import numpy as np

class AutorallyDetector:
    def __init__(self):
        self.svm_model_name = os.path.join('svm.txt')
        svm_vectors = []
        with open('svm.txt') as f:
            for line in f:
                svm_vectors.append(float(line))

        win_size = (128,64)
        block_size = (16,16)
        block_stride = (8,8)
        cell_size = (8,8)
        nbins = 9
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        self.hog.setSVMDetector(np.asarray(svm_vectors))

    def detect(self, img):
        found, w = self.hog.detectMultiScale(im, winStride=(16,16), padding=(8,8), scale=1.05)
        ind = (w > 1)
        w = w[ind]
        found = found[w > 1,:]
        return found

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = 0,0
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

if __name__ == '__main__':
    detector = AutorallyDetector()
    capture = cv2.VideoCapture("/home/igor/py-faster-rcnn/data/demo/autorallyDetection2.mp4")
    cv2.namedWindow('video')
    while True:
        ret, im = capture.read()
        found = detector.detect(im)
        draw_detections(im, found)
        cv2.imshow('video', im)
        if 0xFF & cv2.waitKey(5) == 27:
            break