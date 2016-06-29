import cv2, os
import numpy as np

if __name__ == '__main__':
    hog = cv2.HOGDescriptor((128,64), (16,16), (8,8), (8,8), 9)
    database_path = 'autorally_database'
    save_model_name = os.path.join(database_path, 'svm.dat')
    train_files = os.path.join(database_path, 'ImageSets/Main', 'train.txt')
    test_files = os.path.join(database_path, 'ImageSets/Main', 'test.txt')
    pos_train_files = []
    neg_train_files = []
    pos_test_files = []
    neg_test_files = []
    with open(train_files) as f:
        for line in f:
            index, flag = line.split()
            if flag == '1':
                pos_train_files.append(index)
            elif flag == '-1':
                neg_train_files.append(index)

    with open(test_files) as f:
        for line in f:
            index, flag = line.split()
            if flag == '1':
                pos_test_files.append(index)
            elif flag == '-1':
                neg_test_files.append(index)


    # index = [os.path.splitext(x)[0] for x in files]

    pos_imgs = []
    neg_imgs = []
    for i, file in enumerate(pos_train_files):
        file_name = os.path.join(database_path, 'HOGImages', file + '.jpg')
        pos_imgs.append(cv2.imread(file_name))
    for i, file in enumerate(neg_train_files):
        file_name = os.path.join(database_path, 'HOGImages', file + '.jpg')
        neg_imgs.append(cv2.imread(file_name))

    # Data augmentation
    pos_imgs_aug = []
    neg_imgs_aug = []
    for im in pos_imgs:
        pos_imgs_aug.append(cv2.flip(im, 1))
    for im in neg_imgs:
        neg_imgs_aug.append(cv2.flip(im, 1))

    n_pos = len(pos_imgs) + len(pos_imgs_aug)
    n_neg = len(neg_imgs) + len(neg_imgs_aug)
    pos_features = np.zeros((n_pos, 3780), dtype=np.float32)
    pos_labels = np.zeros(n_pos, dtype=np.int)
    neg_features = np.zeros((n_neg, 3780), dtype=np.float32)
    neg_labels = np.zeros(n_neg, dtype=np.int)

    for i, im in enumerate(pos_imgs + pos_imgs_aug):
        x = np.asarray(hog.compute(im))
        x.astype(np.float32)
        pos_features[i, :] = np.transpose(x)
        pos_labels[i] = 1

    for i, im in enumerate(neg_imgs + neg_imgs_aug):
        x = np.asarray(hog.compute(im))
        x.astype(np.float32)
        neg_features[i, :] = np.transpose(x)
        neg_labels[i] = -1

    params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, term_crit = (cv2.TERM_CRITERIA_MAX_ITER, 50000, 1e-9), C=5, gamma=0.5)
    model = cv2.SVM()
    training_set = np.concatenate((pos_features, neg_features))
    label_set = np.concatenate((pos_labels, neg_labels))

    # model.train(training_set, label_set, params = params)
    model.train(training_set, label_set, params=params)
    model.save('svm.dat')