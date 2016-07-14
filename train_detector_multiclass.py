#!/usr/bin/python

import cv2, os, yaml, sys
import numpy as np

class AutorallyTrainerMultiClass:
    def __init__(self):
        self.database_path = 'autorally_database'
        self.img_path = os.path.join(self.database_path, 'HOGImages')
        self.subcategories_path = os.path.join(self.img_path, 'Subcategories')
        self.save_model_name = os.path.join('svm')
        self.pos_imgs = []
        self.labels = []
        self.neg_imgs = []
        self.imgs = []
        self.svm = cv2.SVM()

        self.gray = True
        #win_size = (128, 64)
        win_size = (96, 48)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        self.subcategories_folders = sorted(os.listdir(self.subcategories_path))
        self.K = len(self.subcategories_folders)

        # positive subcategories
        for i in range(self.K):
            subcategory_path = os.path.join(self.subcategories_path, str(i))
            img_names = sorted(os.listdir(subcategory_path))
            for name in img_names:
                img = cv2.imread(os.path.join(subcategory_path, name))
                if img is not None:
                    self.pos_imgs.append(img)
                    self.labels.append(i)
                else:
                    print "File " + os.path.join(subcategory_path, name) + " not found. Closing program..."
                    sys.exit(0)

        # negative examples
        negatives_path = os.path.join(self.img_path, 'Neg')
        img_names = sorted(os.listdir(negatives_path))
        for name in img_names:
            img = cv2.imread(os.path.join(negatives_path, name))
            if img is not None:
                self.neg_imgs.append(img)
                self.labels.append(self.K)
            else:
                print "File " + os.path.join(negatives_path, name) + " not found. Closing program..."
                sys.exit(0)

        self.imgs = self.pos_imgs + self.neg_imgs

    def train(self):
        feature_size = 1980
        features = np.zeros((len(self.imgs), feature_size), dtype=np.float32)
        labels = np.zeros(len(self.labels), dtype=np.int)
        for i, im in enumerate(self.imgs):
            x = np.asarray(self.hog.compute(im), dtype=np.float32)
            features[i, :] = np.transpose(x)
            labels[i] = self.labels[i]

        params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC,
                      term_crit=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100000, 1e-9), C=2)
        #params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC)

        self.svm.train(features, labels, params=params)
        #self.svm.train_auto(training_set, label_set, None, None, params)
        self.svm.save(self.save_model_name + '.yml')

    def testing(self):
        n_pos_right = 0
        n_pos_wrong = 0
        n_neg_right = 0
        n_neg_wrong = 0
        for i, im in enumerate(self.imgs):
            x = np.asarray(self.hog.compute(im), dtype=np.float32)
            prediction = self.svm.predict(x)
            if self.labels[i] < self.K:
                if prediction < self.K:
                    n_pos_right += 1
                else:
                    n_pos_wrong += 1
            else:
                if prediction == self.K:
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
