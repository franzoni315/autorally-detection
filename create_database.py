#!/usr/bin/python
import cv2, os, random
from xml.dom import minidom

class AutorallyDatabase():
    def __init__(self):
        self.counter = 1
        self.pos_list = []
        self.neg_list = []
        self.voc_database = '/home/igor/Documents/caffe/data/VOCdevkit/VOC2007/'
        self.database_path = 'autorally_database'
        self.win_size = (96,48)

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
                if int(flag) != 1:
                    self.load_negative_annotation(index, self.voc_database, self.database_path, self.neg_list)
                    # self.random_crop(index, self.voc_database, self.database_path, self.neg_list)

        # include hard negatives
        hard_negative_files = os.listdir(os.path.join(self.database_path, 'HardNegativeMining'))
        hard_negative_indices = [os.path.splitext(x)[0] for x in hard_negative_files]
        for index in hard_negative_indices:
            file_name = os.path.join(self.database_path, 'HardNegativeMining', index + '.jpg')
            im = cv2.resize(cv2.imread(file_name), self.win_size)
            gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            save_name = os.path.join(self.database_path, 'HOGImages/Neg', '%05d' % self.counter + '.jpg')
            cv2.imwrite(save_name, gray_img)
            self.counter += 1



        # Generating train and test sets and writing them to files
        train_files = os.path.join(self.database_path, 'ImageSets/Main', 'train.txt')
        test_files = os.path.join(self.database_path, 'ImageSets/Main', 'test.txt')
        random.shuffle(self.pos_list)
        random.shuffle(self.neg_list)
        n_pos_train = int(0.9*len(self.pos_list))
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
        
        xml_name = os.path.join(database_path, 'Annotations', index + '.xml')
        file_name = os.path.join(database_path, 'JPEGImages', index + '.jpg')
        save_name = os.path.join(database_path, 'HOGImages/Pos', '%05d' % self.counter + '.jpg')
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
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop_img = self.crop(gray_img, x1, x2, y1, y2)

        # crop_img = img[y1:y2, x1:x2]
        res_img = cv2.resize(crop_img, self.win_size)
        cv2.imwrite(save_name, res_img)

        save_name = os.path.join(database_path, 'HOGImages/Pos', '%05d' % self.counter + '.jpg')
        index_list.append('%05d' % self.counter)
        self.counter += 1
        flip_img = cv2.flip(res_img, 1)
        cv2.imwrite(save_name, flip_img)

    def load_negative_annotation(self, index, voc_database_path, autorally_database_path, index_list):
        """
        This code is borrowed from Ross Girshick's FAST-RCNN code
        (https://github.com/rbgirshick/fast-rcnn).
        It parses the PASCAL .xml metadata files.
        See publication for further details: (http://arxiv.org/abs/1504.08083).

        Thanks Ross!

        """

        xml_name = os.path.join(voc_database_path, 'Annotations', index + '.xml')
        file_name = os.path.join(voc_database_path, 'JPEGImages', index + '.jpg')
        save_name = os.path.join(autorally_database_path, 'HOGImages/Neg', '%05d' % self.counter + '.jpg')
        index_list.append('%05d' % self.counter)
        self.counter += 1


        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(xml_name) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        img = cv2.imread(file_name)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for obj in objs:
            # Load object bounding boxes into a data frame. Make pixel indexes 0-based
            x1 = (int(get_data_from_tag(obj, 'xmin')) - 1)
            y1 = (int(get_data_from_tag(obj, 'ymin')) - 1)
            x2 = (int(get_data_from_tag(obj, 'xmax')) - 1)
            y2 = (int(get_data_from_tag(obj, 'ymax')) - 1)


            crop_img = gray_img[y1:y2, x1:x2]
            res_img = cv2.resize(crop_img, self.win_size)
            cv2.imwrite(save_name, res_img)
            save_name = os.path.join(autorally_database_path, 'HOGImages/Neg', '%05d' % self.counter + '.jpg')
            index_list.append('%05d' % self.counter)
            self.counter += 1

    def crop(self, img, x1, x2, y1, y2):
        rows, cols = img.shape

        width = x2 - x1
        height = y2 - y1

        dx_r = 0
        dx_l = 0
        dy_u = 0
        dy_d = 0



        while width > 2*height:
            if height % 2 == 0:
                y1 -= 1
            else:
                y2 += 1
            height += 1

        while width < 2*height:
            if width % 2 == 0:
                x1 -= 1
            else:
                x2 += 1
            width += 1


        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
        if x1 < 0:
            pad_left = -x1
            x2 -= x1
            x1 = 0
        if x2 >= cols:
            pad_right = x2 - cols
        if y1 < 0:
            pad_top = -y1
            y2 -= y1
            y1 = 0
        if y2 >= rows:
            pad_bottom = y2 - rows

        aux = cv2.copyMakeBorder( img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE );
        crop_img = aux[y1:y2, x1:x2]
        return crop_img

    def random_crop(self, index, voc_path, autorally_path, index_list):
        file_name = os.path.join(voc_path, 'JPEGImages', index + '.jpg')
        img = cv2.imread(file_name)
        height, width = img.shape[:2]
        win_width = self.win_size[0]
        win_height = self.win_size[1]
        if height > win_height and width > win_width:
            x1 = random.randint(0, width - win_width)
            y1 = random.randint(0, height - win_height)
            x2 = x1 + win_width
            y2 = y1 + win_height
            crop_img = img[y1:y2, x1:x2]
            save_name = os.path.join(self.database_path, 'HOGImages/Neg', '%05d' % self.counter + '.jpg')
            index_list.append('%05d' % self.counter)
            self.counter += 1
            cv2.imwrite(save_name, crop_img)
            

if __name__ == '__main__':
    autorally_database = AutorallyDatabase()
    autorally_database.create_database()





