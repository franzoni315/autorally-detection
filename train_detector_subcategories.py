#!/usr/bin/python

import cv2, os, yaml, sys
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from random import shuffle
from test_detector_subcategories import AutorallyDetectorMultiClass
from xml.dom import minidom

class AutorallyTrainerMultiClass:
    def __init__(self):
        self.voc_database = '/home/igor/Documents/caffe/data/VOCdevkit/VOC2007/'
        self.database_path = 'autorally_database'
        self.img_path = os.path.join(self.database_path, 'HOGImages')
        self.pos_subcategories_path = os.path.join(self.img_path, 'PosSubcategories')
        self.neg_subcategories_path = os.path.join(self.img_path, 'NegSubcategories')
        self.save_model_name = os.path.join('svm')
        self.pos_imgs = []
        self.labels = []
        self.neg_imgs = []
        self.imgs = []
        self.svm_ = SGDClassifier(verbose=False, n_iter=100, n_jobs=8, epsilon=1e-8, loss='log', class_weight='balanced')
        # self.svm_ = svm.LinearSVC(C=4, tol=1e-6, max_iter=1e4, verbose=True)
        # self.svm_ = svm.SVC(C=1, tol=1e-6)
        self.win_size = (96, 48)
        self.block_size = (16, 16)
        self.block_stride = (8, 8)
        self.cell_size = (8, 8)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.nbins)
        self.pos_subcategories_folders = sorted(os.listdir(self.pos_subcategories_path))
        self.Kpos = len(self.pos_subcategories_folders)
        self.neg_subcategories_folders = sorted(os.listdir(self.neg_subcategories_path))
        self.Kneg = len(self.neg_subcategories_folders)
        self.features = []
        self.labels = []

        self.get_database()

    def get_database(self):
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
        self.imgs = self.pos_imgs + self.neg_imgs
        rand_mask = range(len(self.imgs))
        shuffle(rand_mask)
        train_mask = rand_mask[:int(0.9 * len(self.imgs))]
        test_mask = rand_mask[int(0.9 * len(self.imgs)):]
        self.train_imgs = [self.imgs[i] for i in train_mask]
        self.test_imgs = [self.imgs[i] for i in test_mask]
        self.train_labels = [self.labels[i] for i in train_mask]
        self.test_labels = [self.labels[i] for i in test_mask]
        print 'Train set size ', len(self.train_imgs)
        print 'Test set size ', len(self.test_imgs)

    def train(self):
        feature_size = self.hog.getDescriptorSize()
        self.features = np.zeros((len(self.train_imgs), feature_size), dtype=np.float32)
        self.labels = np.zeros(len(self.train_labels), dtype=np.int)
        for i, im in enumerate(self.train_imgs):
            x = np.asarray(self.hog.compute(im), dtype=np.float32)
            self.features[i, :] = np.transpose(x)
            self.labels[i] = self.train_labels[i]
        self.svm_.fit(self.features, self.labels)
        joblib.dump(self.svm_, 'svm.pkl')

        self.testing()

        self.hard_negative_training(self.voc_database)

        self.testing()

        self.hard_negative_training(self.database_path)

        self.testing()

    def hard_negative_training(self, database_path):
        detector = AutorallyDetectorMultiClass('svm.pkl')
        negatives_list = []
        negatives_labels = []
        negative_filename = os.path.join(database_path, 'ImageSets/Main', 'car_trainval.txt')
        with open(negative_filename, "r") as f:
            for line in f:
                index, flag = line.split()
                print index
                img, gt_boxes = self.load_pascal_annotation(index, database_path)

                xscale = float(96*8.0/img.shape[1])
                yscale = float(48*12.0/img.shape[0])
                for box in gt_boxes:
                    box[0] = int(box[0]*xscale)
                    box[2] = int(box[2]*xscale)
                    box[1] = int(box[1]*yscale)
                    box[3] = int(box[3]*yscale)
                img = cv2.resize(img, (96*8, 48*12))
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                found, w = detector.detectMultiScale(img)
                for box in found:
                    prediction_correct = False
                    for gt_box in gt_boxes:
                        iou = self.inter_over_union([box[0], box[1], box[0]+box[2], box[1]+box[3]], gt_box)
                        if iou > 0.35:
                            prediction_correct = True
                            continue
                    if not prediction_correct:
                        crop_img = gray_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                        res_img = cv2.resize(crop_img, self.win_size)
                        negatives_list.append(res_img)
                        negatives_labels.append(self.Kpos)

        feature_size = self.hog.getDescriptorSize()
        features = np.zeros((len(negatives_list), feature_size), dtype=np.float32)
        labels = np.zeros(len(negatives_labels), dtype=np.int)
        if len(negatives_list) != 0:
            for i, im in enumerate(negatives_list):
                x = np.asarray(self.hog.compute(im), dtype=np.float32)
                features[i, :] = np.transpose(x)
                labels[i] = negatives_labels[i]
            self.features = np.concatenate((self.features, features))
            self.labels = np.concatenate((self.labels, labels))
            self.svm_.fit(self.features, self.labels)
            joblib.dump(self.svm_, 'svm_fine_tune.pkl')

    def inter_over_union(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        w = x2-x1
        h = y2-y1
        inter = w*h
        aarea = (a[2]-a[0]) * (a[3]-a[1])
        barea = (b[2]-b[0]) * (b[3]-b[1])
        iou = 1.0* inter / (aarea+barea-inter)
        if w <= 0 or h <= 0:
            return 0
        else:
            return iou


    def load_pascal_annotation(self, index, database_path):
        xml_name = os.path.join(database_path, 'Annotations', index + '.xml')
        file_name = os.path.join(database_path, 'JPEGImages', index + '.jpg')

        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(xml_name) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        img = cv2.imread(file_name)
        car_boxes = []
        for obj in objs:
            # Load object bounding boxes into a data frame. Make pixel indexes 0-based
            if (get_data_from_tag(obj, 'name')) == 'car':
                x1 = (int(get_data_from_tag(obj, 'xmin')) - 1)
                y1 = (int(get_data_from_tag(obj, 'ymin')) - 1)
                x2 = (int(get_data_from_tag(obj, 'xmax')) - 1)
                y2 = (int(get_data_from_tag(obj, 'ymax')) - 1)
                car_boxes.append([x1,y1,x2,y2])
        return img, car_boxes

    def testing(self):
        n_pos_right = 0
        n_pos_wrong = 0
        n_neg_right = 0
        n_neg_wrong = 0
        for i, im in enumerate(self.test_imgs):
            x = np.asarray(self.hog.compute(im), dtype=np.float32)
            score = self.svm_.predict_proba(np.transpose(x))
            pos_prob = np.sum(score[0,:self.Kpos])
            neg_prob = np.sum(score[0, self.Kpos:])

            # prediction = self.svm_.predict(np.transpose(x))
            if self.test_labels[i] < self.Kpos:
                if pos_prob > neg_prob:
                    n_pos_right += 1
                else:
                    n_pos_wrong += 1
            else:
                if pos_prob < neg_prob:
                    n_neg_right += 1
                else:
                    n_neg_wrong += 1

        print 'Confusion matrix on test set:'
        print '%4d' % n_pos_right, '%4d' % n_neg_wrong
        print '%4d' % n_pos_wrong, '%4d' % n_neg_right

if __name__ == '__main__':
    trainer = AutorallyTrainerMultiClass()
    trainer.train()
