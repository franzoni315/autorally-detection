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
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, w = self.hog.detectMultiScale(gray_img, winStride=(16,16), padding=(8,8), scale=1.05)
        #ind = (w > 1)
        #w = w[ind]
        #found = found[w > 1,:]
        print w
        return found

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness)

if __name__ == '__main__':
    detector = AutorallyDetector()
    capture = cv2.VideoCapture("/home/igor/Documents/autorally-detection/autorally_database/Videos/0002.mp4")
    # capture = cv2.VideoCapture("/home/igor/Downloads/test.mp4")

    cv2.namedWindow('video')
    while True:
        ret, im = capture.read()
        im = cv2.resize(im, (640, 480))
        found = detector.detect(im)
        draw_detections(im, found)
        cv2.imshow('video', im)
        if 0xFF & cv2.waitKey(5) == 27:
            break
