import cv2, os, yaml
import numpy as np


class AutorallyTrainer:
    def __init__(self):
        self.database_path = 'autorally_database'
        self.save_model_name = os.path.join('svm.txt')
        self.train_path = os.path.join(self.database_path, 'ImageSets/Main', 'train.txt')
        self.test_path = os.path.join(self.database_path, 'ImageSets/Main', 'test.txt')
        self.pos_train_imgs = []
        self.neg_train_imgs = []
        self.pos_test_imgs = []
        self.neg_test_imgs = []
        self.svm = cv2.SVM()
        self.gray = True
        win_size = (128, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        with open(self.train_path) as f:
            for line in f:
                index, flag = line.split()
                if flag == '1':
                    file_name = os.path.join(self.database_path, 'HOGImages', index + '.jpg')
                    im = cv2.imread(file_name)
                    if self.gray:
                        self.pos_train_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
                    else:
                        self.pos_train_imgs.append(im)
                elif flag == '-1':
                    file_name = os.path.join(self.database_path, 'HOGImages', index + '.jpg')
                    im = cv2.imread(file_name)
                    if self.gray:
                        self.neg_train_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
                    else:
                        self.neg_train_imgs.append(im)

        with open(self.test_path) as f:
            for line in f:
                index, flag = line.split()
                if flag == '1':
                    file_name = os.path.join(self.database_path, 'HOGImages', index + '.jpg')
                    im = cv2.imread(file_name)
                    if self.gray:
                        self.pos_test_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
                    else:
                        self.pos_test_imgs.append(im)
                elif flag == '-1':
                    file_name = os.path.join(self.database_path, 'HOGImages', index + '.jpg')
                    im = cv2.imread(file_name)
                    if self.gray:
                        self.neg_test_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
                    else:
                        self.neg_test_imgs.append(im)

        # include hard negatives
        hard_negative_files = os.listdir(os.path.join(self.database_path, 'HardNegativeMining'))
        hard_negative_indices = [os.path.splitext(x)[0] for x in hard_negative_files]
        for index in hard_negative_indices:
            file_name = os.path.join(self.database_path, 'HardNegativeMining', index + '.jpg')
            im = cv2.imread(file_name)
            if self.gray:
                self.neg_train_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            else:
                self.neg_train_imgs.append(im)


    def data_augmentation(self):
        # Data augmentation
        pos_imgs_aug = []
        neg_imgs_aug = []
        for im in self.pos_train_imgs:
            pos_imgs_aug.append(cv2.flip(im, 1))

            rows,cols = im.shape
            M1 = cv2.getRotationMatrix2D((cols/2,rows/2),5,1)
            M2 = cv2.getRotationMatrix2D((cols/2,rows/2),-5,1)
            dst1 = cv2.warpAffine(im,M1,(cols,rows))
            dst2 = cv2.warpAffine(im,M2,(cols,rows))
            pos_imgs_aug.append(cv2.resize(dst1,(128,64)))
            pos_imgs_aug.append(cv2.resize(dst2,(128,64)))
        for im in self.neg_train_imgs:
            neg_imgs_aug.append(cv2.flip(im, 1))

        self.pos_train_imgs += pos_imgs_aug
        self.neg_train_imgs += neg_imgs_aug



    def train(self):
        n_pos = len(self.pos_train_imgs)
        n_neg = len(self.neg_train_imgs)
        pos_features = np.zeros((n_pos, 3780), dtype=np.float32)
        pos_labels = np.zeros(n_pos, dtype=np.int)
        neg_features = np.zeros((n_neg, 3780), dtype=np.float32)
        neg_labels = np.zeros(n_neg, dtype=np.int)

        for i, im in enumerate(self.pos_train_imgs):
            x = np.asarray(self.hog.compute(im), dtype=np.float32)
            pos_features[i, :] = np.transpose(x)
            pos_labels[i] = 1

        for i, im in enumerate(self.neg_train_imgs):
            x = np.asarray(self.hog.compute(im), dtype=np.float32)
            neg_features[i, :] = np.transpose(x)
            neg_labels[i] = -1

        params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC,
                      term_crit=(cv2.TERM_CRITERIA_MAX_ITER or cv2.TERM_CRITERIA_EPS, 100000, 1e-9), C=1, gamma=0,
                      p=0.1)

        training_set = np.concatenate((pos_features, neg_features))
        label_set = np.concatenate((pos_labels, neg_labels))
        self.svm.train(training_set, label_set, params=params)
        self.svm.save(self.save_model_name)
        self.format_output()

    def format_output(self):
        skip_lines = 15
        with open(self.save_model_name, 'r') as f:
            for i in range(skip_lines):
                dummy = f.readline()
            data = yaml.load(f)
        svm_vectors = data['support_vectors'][0]
        svm_vectors = [-x for x in svm_vectors]
        rho = data['decision_functions'][0]['rho']
        svm_vectors.append(rho)

        print 'SVM weights: \n', svm_vectors, '\n'
        with open(self.save_model_name, 'w') as f:
            for elem in svm_vectors:
                f.write(str(elem) + '\n')

    def testing(self):
        n_pos = len(self.pos_test_imgs)
        n_pos_right = 0
        n_pos_wrong = 0
        for img in self.pos_test_imgs:
            features = self.hog.compute(img)
            prediction = self.svm.predict(features)
            if prediction == -1:
                n_pos_wrong += 1
            elif prediction == 1:
                n_pos_right += 1

        n_neg = len(self.neg_test_imgs)
        n_neg_right = 0
        n_neg_wrong = 0
        for img in self.neg_test_imgs:
            features = self.hog.compute(img)
            prediction = self.svm.predict(features)
            if prediction == 1:
                n_neg_wrong += 1
            elif prediction == -1:
                n_neg_right += 1

        print 'Confusion matrix:'
        print '%4d' % n_pos_right, '%4d' % n_neg_wrong
        print '%4d' % n_pos_wrong, '%4d' % n_neg_right


if __name__ == '__main__':
    trainer = AutorallyTrainer()
    trainer.data_augmentation()
    trainer.train()
    trainer.testing()
