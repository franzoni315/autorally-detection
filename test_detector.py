import cv2, os, yaml,sys,re
import numpy as np

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = 0,0
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

if __name__ == '__main__':
    capture = cv2.VideoCapture("/home/igor/py-faster-rcnn/data/demo/autorallyDetection2.mp4")
    cv2.namedWindow('video')
    svmvec = []
    with open('svm.txt') as f:
        svmvec = filter(None, re.split("[, \n]", f.read()))
    svmvec = [-float(x) for x in svmvec]
    svmvec[-1] = -svmvec[-1]

    hog = cv2.HOGDescriptor((128,64), (16,16), (8,8), (8,8), 9)
    hog.setSVMDetector(np.array(svmvec))
    while True:
        ret, im = capture.read()
        # gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        found, w = hog.detectMultiScale(im, winStride=(16,16), padding=(8,8), scale=1.05)
        ind = (w > 1)
        w = w[ind]
        found = found[w > 1,:]
        # max = np.argmax(w)
        # found=[found[max,:]]
        print w, found
        # found_filtered = []
        # for ri, r in enumerate(found):
        #     for qi, q in enumerate(found):
        #         if ri != qi and inside(r, q):
        #             break
        #     else:
        #         found_filtered.append(r)
        draw_detections(im, found)
        # draw_detections(im, found_filtered, 3)
        cv2.imshow('video', im)
        if 0xFF & cv2.waitKey(5) == 27:
            break

    # im = cv2.imread('autorally_database/JPEGImages/00020.jpg')
    # gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # svmvec = []
    # with open('svm.txt') as f:
    #     svmvec = filter(None, re.split("[, \n]", f.read()))
    # svmvec = [-float(x) for x in svmvec]
    # svmvec[-1] = -svmvec[-1]
    #
    # hog = cv2.HOGDescriptor((128,64), (16,16), (8,8), (8,8), 9)
    # hog.setSVMDetector(np.array(svmvec))
    # found, w = hog.detectMultiScale(gray_im, winStride=(16,16), padding=(32,32), scale=1.1)
    # found_filtered = []
    # for ri, r in enumerate(found):
    #     for qi, q in enumerate(found):
    #         if ri != qi and inside(r, q):
    #             break
    #     else:
    #         found_filtered.append(r)
    # draw_detections(im, found)
    # draw_detections(im, found_filtered, 3)
    # print '%d (%d) found' % (len(found_filtered), len(found))
    # cv2.imshow('img', im)
    # cv2.waitKey(0)