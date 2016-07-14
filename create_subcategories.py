#!/usr/bin/python
import cv2, os, sys, shutil
import numpy as np

database_path = 'autorally_database'
kmeans_path = os.path.join(database_path, 'HOGImages/Subcategories')
im_path = os.path.join(database_path, 'HOGImages/Pos')
files = sorted(os.listdir(im_path))

win_size = (96, 48)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9
feature_size = 1980
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

# remove old folders in kmeans_path
for the_file in os.listdir(kmeans_path):
    file_path = os.path.join(kmeans_path, the_file)
    try:
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(e)

imgs = []
for i, file in enumerate(files):
    img_aux = cv2.imread(os.path.join(im_path, file))
    imgs.append(cv2.cvtColor(img_aux, cv2.COLOR_BGR2GRAY))


hog_features = np.zeros((len(imgs), feature_size), dtype=np.float32)

for i, im in enumerate(imgs):
    x = np.asarray(hog.compute(im), dtype=np.float32)
    hog_features[i, :] = np.transpose(x)

K = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-8)
attempts = 10
compactness, bestLabels, centers = cv2.kmeans(hog_features, K, criteria, attempts, cv2.KMEANS_PP_CENTERS)
print compactness
print bestLabels
print centers

for i in range(K):
    os.makedirs(os.path.join(kmeans_path, str(i)))

for i, img in enumerate(imgs):
    print os.path.join(kmeans_path, str(bestLabels[i][0]), files[i])
    cv2.imwrite(os.path.join(kmeans_path, str(bestLabels[i][0]), files[i]), img)