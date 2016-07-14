#!/usr/bin/python
import cv2, os, sys
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Displays the mean image of all images of a folder.')
parser.add_argument('path', help='Folder path')
parser.add_argument('savename', help='Mean image filename.')
args = parser.parse_args()

path = args.path
savename = args.savename

# Check if path exits
if os.path.exists(path):
    print "Renaming all *.jpg files in folder ", path
else:
    print "Path not found. Closing program..."
    sys.exit(0)

filenames = os.listdir(path)
filenames.sort()
cv2.namedWindow('mean image')

mean_image = np.zeros((48,96), np.float32)

i = 0
for filename in filenames:
    title, ext = os.path.splitext(os.path.basename(filename))
    print filename
    if ext == '.jpg':
        im = cv2.imread(path + filename)
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_im = gray_im.astype(np.float32)
        gray_im /= 255.0
        mean_image += gray_im
        i += 1

mean_image = mean_image/i*255
mean_image = mean_image.astype(np.uint8)
cv2.imwrite(savename + '.jpg', mean_image)

        
    
