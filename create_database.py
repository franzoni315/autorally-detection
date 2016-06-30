import cv2, os, random
from xml.dom import minidom

class AutorallyDatabase():
    def __init__(self):
        self.counter = 1
        self.pos_list = []
        self.neg_list = []
        self.voc_database = '/home/igor/Documents/caffe/data/VOCdevkit/VOC2007/'
        self.database_path = 'autorally_database'

    def create_database(self):
        # Generating positive examples
        files = sorted(os.listdir(os.path.join(self.database_path, 'Annotations')))
        index = [os.path.splitext(x)[0] for x in files]
        for ix in index:
            self.load_pascal_annotation(ix, self.database_path, self.pos_list)
        # Generating negative examples
        negative_filename = os.path.join(self.voc_database, 'ImageSets/Main', 'car_trainval.txt')
        with open(negative_filename, "r") as f:
            for line in f:
                index, flag = line.split()
                if flag != 1:
                    self.random_crop(index, self.voc_database, self.database_path, self.neg_list)

        # Generating train and test sets and writing them to files
        train_files = os.path.join(self.database_path, 'ImageSets/Main', 'train.txt')
        test_files = os.path.join(self.database_path, 'ImageSets/Main', 'test.txt')
        random.shuffle(self.pos_list)
        random.shuffle(self.neg_list)
        n_pos_train = int(0.75*len(self.pos_list))
        n_neg_train = int(0.75*len(self.neg_list))
        pos_train = self.pos_list[0:n_pos_train]
        pos_test = self.pos_list[n_pos_train:]
        neg_train = self.neg_list[0:n_neg_train]
        neg_test = self.neg_list[n_neg_train:]
        with open(train_files, 'w') as f:
            for ix in pos_train:
                f.write(ix + ' 1\n')
            for ix in neg_train:
                f.write(ix + ' -1\n')
        with open(test_files, 'w') as f:
            for ix in pos_test:
                f.write(ix + ' 1\n')
            for ix in neg_test:
                f.write(ix + ' -1\n')


    def load_pascal_annotation(self, index, database_path, index_list):
        """
        This code is borrowed from Ross Girshick's FAST-RCNN code
        (https://github.com/rbgirshick/fast-rcnn).
        It parses the PASCAL .xml metadata files.
        See publication for further details: (http://arxiv.org/abs/1504.08083).

        Thanks Ross!

        """
        hog_width = 128
        hog_height = 64
        xml_name = os.path.join(database_path, 'Annotations', index + '.xml')
        file_name = os.path.join(database_path, 'JPEGImages', index + '.jpg')
        save_name = os.path.join(database_path, 'HOGImages', '%05d' % self.counter + '.jpg')
        index_list.append('%05d' % self.counter)
        self.counter += 1

        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(xml_name) as f:
            data = minidom.parseString(f.read())

        obj = data.getElementsByTagName('object')[0]

        # Load object bounding boxes into a data frame. Make pixel indexes 0-based
        x1 = (int(get_data_from_tag(obj, 'xmin')) - 1)
        y1 = (int(get_data_from_tag(obj, 'ymin')) - 1)
        x2 = (int(get_data_from_tag(obj, 'xmax')) - 1)
        y2 = (int(get_data_from_tag(obj, 'ymax')) - 1)

        img = cv2.imread(file_name)
        crop_img = img[y1:y2, x1:x2]
        res_img = cv2.resize(crop_img,(hog_width,hog_height))
        cv2.imwrite(save_name, res_img)

    def random_crop(self, index, voc_path, autorally_path, index_list):
        file_name = os.path.join(voc_path, 'JPEGImages', index + '.jpg')
        img = cv2.imread(file_name)
        height, width = img.shape[:2]
        if height > 64 and width > 128:
            x1 = random.randint(0, width - 128)
            y1 = random.randint(0, height - 64)
            x2 = x1 + 128
            y2 = y1 + 64
            crop_img = img[y1:y2, x1:x2]
            save_name = os.path.join(self.database_path, 'HOGImages', '%05d' % self.counter + '.jpg')
            index_list.append('%05d' % self.counter)
            self.counter += 1
            cv2.imwrite(save_name, crop_img)

if __name__ == '__main__':
    autorally_database = AutorallyDatabase()
    autorally_database.create_database()





