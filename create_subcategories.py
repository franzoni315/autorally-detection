#!/usr/bin/python
import cv2
import os
import shutil
import numpy as np

class ImageClustering():
    def __init__(self, kmeans_path, im_path, K):
        self.database_path = 'autorally_database'
        self.kmeans_path = kmeans_path
        self.im_path = im_path
        self.files = sorted(os.listdir(self.im_path))
        self.win_size = (96, 48)
        self.block_size = (16, 16)
        self.block_stride = (8, 8)
        self.cell_size = (8, 8)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.nbins)
        self.feature_size = self.hog.getDescriptorSize()
        self.K = K
        if os.path.isdir(kmeans_path):
            print 'Removing {} folder...'.format(kmeans_path)
            shutil.rmtree(kmeans_path)
        os.makedirs(kmeans_path)
    def cluster(self):
        imgs = []
        for i, file in enumerate(self.files):
            img_aux = cv2.imread(os.path.join(self.im_path, file))
            gray_img = cv2.cvtColor(img_aux, cv2.COLOR_BGR2GRAY)
            imgs.append(gray_img)

        hog_features = np.zeros((len(imgs), self.feature_size), dtype=np.float32)

        for i, im in enumerate(imgs):
            x = np.asarray(self.hog.compute(im, self.win_size), dtype=np.float32)
            hog_features[i, :] = np.transpose(x)


        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100000, 1e-8)
        attempts = 50
        compactness, bestLabels, centers = cv2.kmeans(hog_features, self.K, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

        for i in range(self.K):
            os.makedirs(os.path.join(self.kmeans_path, str(i)))

        for i, img in enumerate(imgs):
            print os.path.join(self.kmeans_path, str(bestLabels[i][0]), self.files[i])
            cv2.imwrite(os.path.join(self.kmeans_path, str(bestLabels[i][0]), self.files[i]), img)

if __name__ == '__main__':
    c = ImageClustering('database/NegSubcategories', 'database/Neg', 1)
    c.cluster()
    c = ImageClustering('database/PosSubcategories', 'database/Pos', 8)
    c.cluster()