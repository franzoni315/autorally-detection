#!/usr/bin/python
import cv2
import os
import random
import  shutil
from xml.dom import minidom

class AutorallyDatabase():
    def __init__(self):
        self.counter = 1
        self.pos_list = []
        self.neg_list = []
        self.voc_database = '/home/igor/Documents/caffe/data/VOCdevkit/VOC2007/'
        self.database_path = 'autorally_database'
        self.win_size = (96,48)
        if os.path.isdir('database'):
            print 'Removing "database" folder...'
            shutil.rmtree('database')
        print 'Creating a new "database" folder...'
        os.makedirs('database/Pos')
        os.makedirs('database/Neg')

    def create_database(self):
        # Generating positive examples
        files = sorted(os.listdir(os.path.join(self.database_path, 'Annotations')))
        index = [os.path.splitext(x)[0] for x in files]
        train_files = os.path.join(self.database_path, 'ImageSets/Main', 'car_trainval.txt')
        with open(train_files, 'w') as f:
            for ix in index:
                self.load_pascal_annotation(ix, self.database_path, self.pos_list)
                f.write(ix + ' 1\n')
        # Generating negative examples
        negative_filename = os.path.join(self.voc_database, 'ImageSets/Main', 'car_trainval.txt')
        with open(negative_filename, "r") as f:
            for line in f:
                index, flag = line.split()
                if int(flag) != 1:
                    self.load_negative_annotation(index, self.voc_database, self.database_path, self.neg_list)

        # include hard negatives
        hard_negative_files = os.listdir(os.path.join(self.database_path, 'HardNegativeMining'))
        hard_negative_indices = [os.path.splitext(x)[0] for x in hard_negative_files]
        for index in hard_negative_indices:
            file_name = os.path.join(self.database_path, 'HardNegativeMining', index + '.jpg')
            im = cv2.resize(cv2.imread(file_name), self.win_size)
            gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            save_name = os.path.join('database/Neg', '%05d' % self.counter + '.jpg')
            cv2.imwrite(save_name, gray_img)
            self.counter += 1



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
        save_name = os.path.join('database/Pos', '%05d' % self.counter + '.jpg')
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

        save_name = os.path.join('database/Pos', '%05d' % self.counter + '.jpg')
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
        save_name = os.path.join('database/Neg', '%05d' % self.counter + '.jpg')
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
            save_name = os.path.join('database/Neg', '%05d' % self.counter + '.jpg')
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

if __name__ == '__main__':
    autorally_database = AutorallyDatabase()
    autorally_database.create_database()





