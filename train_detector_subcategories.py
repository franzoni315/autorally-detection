#!/usr/bin/python

import cv2, os, yaml, sys
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from random import shuffle

class AutorallyTrainerMultiClass:
    def __init__(self):
        self.database_path = 'autorally_database'
        self.img_path = os.path.join(self.database_path, 'HOGImages')
        self.pos_subcategories_path = os.path.join(self.img_path, 'PosSubcategories')
        self.neg_subcategories_path = os.path.join(self.img_path, 'NegSubcategories')
        self.save_model_name = os.path.join('svm')
        self.pos_imgs = []
        self.labels = []
        self.neg_imgs = []
        self.imgs = []
        self.svm_ = SGDClassifier(verbose=True, n_iter=20, n_jobs=4, epsilon=1e-6, loss='log')
        # self.svm_ = svm.LinearSVC(C=4, tol=1e-6, max_iter=1e4, verbose=True)
        # self.svm_ = svm.SVC(C=1, tol=1e-6)

        self.gray = True
        win_size = (96, 48)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        self.pos_subcategories_folders = sorted(os.listdir(self.pos_subcategories_path))
        self.Kpos = len(self.pos_subcategories_folders)
        self.neg_subcategories_folders = sorted(os.listdir(self.neg_subcategories_path))
        self.Kneg = len(self.neg_subcategories_folders)


        # positive subcategories
        for i in range(self.Kpos):
            subcategory_path = os.path.join(self.pos_subcategories_path, str(i))
            print subcategory_path
            img_names = sorted(os.listdir(subcategory_path))
            for name in img_names:
                img = cv2.imread(os.path.join(subcategory_path, name))
                if img is not None:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self.pos_imgs.append(img_gray)
                    self.labels.append(i)
                else:
                    print "File " + os.path.join(subcategory_path, name) + " not found. Closing program..."
                    sys.exit(0)

        # negative examples
        for i in range(self.Kneg):
            subcategory_path = os.path.join(self.neg_subcategories_path, str(i))
            img_names = sorted(os.listdir(subcategory_path))
            for name in img_names:
                img = cv2.imread(os.path.join(subcategory_path, name))
                if img is not None:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self.neg_imgs.append(img_gray)
                    self.labels.append(self.Kpos + i)
                else:
                    print "File " + os.path.join(subcategory_path, name) + " not found. Closing program..."
                    sys.exit(0)

        # negatives_path = os.path.join(self.img_path, 'Neg')
        # img_names = sorted(os.listdir(negatives_path))
        # for name in img_names:
        #     img = cv2.imread(os.path.join(negatives_path, name))
        #     if img is not None:
        #         self.neg_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        #         self.labels.append(self.Kpos)
        #     else:
        #         print "File " + os.path.join(negatives_path, name) + " not found. Closing program..."
        #         sys.exit(0)

        # include hard negatives
        # hard_negative_files = os.listdir(os.path.join(self.database_path, 'HardNegativeMining'))
        # hard_negative_indices = [os.path.splitext(x)[0] for x in hard_negative_files]
        # for index in hard_negative_indices:
        #     file_name = os.path.join(self.database_path, 'HardNegativeMining', index + '.jpg')
        #     im = cv2.resize(cv2.imread(file_name), win_size)
        #     if self.gray:
        #         self.neg_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        #         self.labels.append(self.Kpos)

        self.imgs = self.pos_imgs + self.neg_imgs
        rand_mask = range(len(self.imgs))
        shuffle(rand_mask)
        train_mask = rand_mask[:int(0.8*len(self.imgs))]
        test_mask = rand_mask[int(0.8*len(self.imgs)):]
        self.train_imgs = [self.imgs[i] for i in train_mask]
        self.test_imgs = [self.imgs[i] for i in test_mask]
        self.train_labels = [self.labels[i] for i in train_mask]
        self.test_labels = [self.labels[i] for i in test_mask]

        print 'Train set size ', len(self.train_imgs)
        print 'Test set size ', len(self.test_imgs)

    def train(self):
        feature_size = self.hog.getDescriptorSize()
        features = np.zeros((len(self.train_imgs), feature_size), dtype=np.float32)
        labels = np.zeros(len(self.train_labels), dtype=np.int)
        for i, im in enumerate(self.train_imgs):
            x = np.asarray(self.hog.compute(im), dtype=np.float32)
            features[i, :] = np.transpose(x)
            labels[i] = self.train_labels[i]

        self.svm_.fit(features, labels)
        joblib.dump(self.svm_, 'svm.pkl')

    def testing(self):
        n_pos_right = 0
        n_pos_wrong = 0
        n_neg_right = 0
        n_neg_wrong = 0
        for i, im in enumerate(self.test_imgs):
            x = np.asarray(self.hog.compute(im), dtype=np.float32)
            prediction = self.svm_.predict(np.transpose(x))
            print prediction
            if self.test_labels[i] < self.Kpos:
                if prediction < self.Kpos:
                    n_pos_right += 1
                else:
                    n_pos_wrong += 1
            else:
                if prediction >= self.Kpos:
                    n_neg_right += 1
                else:
                    n_neg_wrong += 1

        print 'Confusion matrix on test set:'
        print '%4d' % n_pos_right, '%4d' % n_neg_wrong
        print '%4d' % n_pos_wrong, '%4d' % n_neg_right



if __name__ == '__main__':
    trainer = AutorallyTrainerMultiClass()
    trainer.train()
    trainer.testing()
