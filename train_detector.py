#!/usr/bin/python
import sys
sys.path.insert(1, "/home/igor/Documents/opencv-3.1.0/build/lib")
import cv2
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from random import shuffle
from test_detector import AutorallyDetectorMultiClass
from xml.dom import minidom

class AutorallyTrainerMultiClass:
    def __init__(self):
        self.voc_database = '/home/igor/Documents/caffe/data/VOCdevkit/VOC2007/'
        self.database_path = 'autorally_database'
        self.pos_subcategories_path = os.path.join('database/PosSubcategories')
        self.neg_subcategories_path = os.path.join('database/NegSubcategories')
        self.svm_ = SGDClassifier(verbose=True, n_iter=200, n_jobs=8, epsilon=1e-8, loss='hinge', class_weight='balanced', warm_start=True)
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
        self.train_features = []
        self.train_labels = []
        self.test_features = []
        self.test_labels = []
        self.class_of_interest = 'car'

        self.get_database()

    def get_database(self):
        imgs = []
        labels = []
        # positive subcategories
        for i in range(self.Kpos):
            subcategory_path = os.path.join(self.pos_subcategories_path, str(i))
            print subcategory_path
            img_names = sorted(os.listdir(subcategory_path))
            for name in img_names:
                img = cv2.imread(os.path.join(subcategory_path, name))
                if img is not None:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    imgs.append(img_gray)
                    labels.append(i)
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
                    imgs.append(img_gray)
                    labels.append(self.Kpos + i)
                else:
                    print "File " + os.path.join(subcategory_path, name) + " not found. Closing program..."
                    sys.exit(0)

        rand_mask = range(len(imgs))
        shuffle(rand_mask)
        train_mask = rand_mask[:int(0.9 * len(imgs))]
        test_mask = rand_mask[int(0.9 * len(imgs)):]
        train_imgs = [imgs[i] for i in train_mask]
        test_imgs = [imgs[i] for i in test_mask]
        train_labels = [labels[i] for i in train_mask]
        test_labels = [labels[i] for i in test_mask]

        feature_size = self.hog.getDescriptorSize()
        self.train_features = np.zeros((len(train_imgs), feature_size), dtype=np.float32)
        self.train_labels = np.zeros(len(train_labels), dtype=np.int)
        for i, im in enumerate(train_imgs):
            x = np.asarray(self.hog.compute(im), dtype=np.float32)
            self.train_features[i, :] = np.transpose(x)
            self.train_labels[i] = train_labels[i]

        self.test_features = np.zeros((len(test_imgs), feature_size), dtype=np.float32)
        self.test_labels = np.zeros(len(test_labels), dtype=np.int)
        for i, im in enumerate(test_imgs):
            x = np.asarray(self.hog.compute(im), dtype=np.float32)
            self.test_features[i, :] = np.transpose(x)
            self.test_labels[i] = test_labels[i]

        print 'Train set size ', len(self.train_features)
        print 'Test set size ', len(self.test_features)

    def train(self):
        print 'Training with ', self.train_features.shape[0], ' features...'
        self.svm_.fit(self.train_features, self.train_labels)
        joblib.dump(self.svm_, 'svm.pkl', compress=1)
        self.testing()
        for i in range(2):
            self.hard_negative_training(self.database_path, 'svm.pkl', 'svm.pkl')
            self.testing()

        # self.hard_negative_training(self.voc_database, 'svm1.pkl', 'svm2.pkl')
        # self.testing()

    def hard_negative_training(self, database_path, svm, save):
        detector = AutorallyDetectorMultiClass(svm)
        negatives_list = []
        negatives_labels = []

        files = sorted(os.listdir(os.path.join(database_path, 'Annotations')))
        index = [os.path.splitext(x)[0] for x in files]
        for ix in index:
            print ix
            img, gt_boxes = self.load_pascal_annotation(ix, database_path)

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

            self.train_features = np.concatenate((self.train_features, features))
            self.train_labels = np.concatenate((self.train_labels, labels))
            del negatives_labels, negatives_list, features, labels
            print 'Training with ', self.train_features.shape[0], ' features...'
            self.svm_.fit(self.train_features, self.train_labels)
            joblib.dump(self.svm_, save, compress=1)

    def inter_over_union(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        w = x2-x1
        h = y2-y1
        if w <= 0 or h <= 0:
            return 0
        inter = w*h
        aarea = (a[2]-a[0]) * (a[3]-a[1])
        barea = (b[2]-b[0]) * (b[3]-b[1])
        iou = 1.0* inter / (aarea+barea-inter)

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
            if (get_data_from_tag(obj, 'name')) == self.class_of_interest:
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
        for i, im in enumerate(self.test_features):
            score = self.svm_.decision_function(np.reshape(self.test_features[i, :], (1,self.hog.getDescriptorSize())))[0]
            neg_weight = 0
            for j in range(self.Kpos, self.Kpos + self.Kneg):
                if score[j] > 0:
                    neg_weight += score[j]
            pos_weight = 0
            for j in range(self.Kpos):
                if score[j] > 0:
                    pos_weight += score[j]

            if self.test_labels[i] < self.Kpos:
                if pos_weight > neg_weight:
                    n_pos_right += 1
                else:
                    n_pos_wrong += 1
            else:
                if pos_weight < neg_weight:
                    n_neg_right += 1
                else:
                    n_neg_wrong += 1

        print 'Confusion matrix on test set:'
        print '%4d' % n_pos_right, '%4d' % n_neg_wrong
        print '%4d' % n_pos_wrong, '%4d' % n_neg_right

if __name__ == '__main__':
    trainer = AutorallyTrainerMultiClass()
    trainer.train()
