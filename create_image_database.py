import cv2, os, random
from xml.dom import minidom



def load_pascal_annotation(index, database_path, index_list, counter):
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
    save_name = os.path.join(database_path, 'HOGImages', '%05d' % counter + '.jpg')
    index_list.append('%05d' % counter)
    counter += 1
    # save_name_flipped = os.path.join(database_path, 'HOGImages', '%05d' % counter + '.jpg')
    # index_list.append('%05d' % counter)
    # counter += 1

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
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crop_img = img[y1:y2, x1:x2]
    res_img = cv2.resize(crop_img,(hog_width,hog_height))
    flip_img = cv2.flip(res_img, 1)
    cv2.imwrite(save_name, res_img)
    # cv2.imwrite(save_name_flipped, flip_img)
    return counter

def random_crop(index, voc_path, autorally_path, index_list, counter):
    file_name = os.path.join(voc_path, 'JPEGImages', index + '.jpg')

    img = cv2.imread(file_name)
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    if height > 64 and width > 128:
        x1 = random.randint(0, width - 128)
        y1 = random.randint(0, height - 64)
        x2 = x1 + 128
        y2 = y1 + 64
        crop_img = img[y1:y2, x1:x2]
        save_name = os.path.join(autorally_database_path, 'HOGImages', '%05d' % counter + '.jpg')
        index_list.append('%05d' % counter)
        counter += 1
        cv2.imwrite(save_name, crop_img)
    return counter

if __name__ == '__main__':
    counter = 1
    pos_list = []
    neg_list = []
    # Generating positive examples
    voc_database = '/home/igor/Documents/caffe/data/VOCdevkit/VOC2007/'
    autorally_database_path = 'autorally_database'
    files = sorted(os.listdir(os.path.join(autorally_database_path, 'Annotations')))
    index = [os.path.splitext(x)[0] for x in files]
    for ix in index:
        counter = load_pascal_annotation(ix, autorally_database_path, pos_list, counter)
    # Generating negative examples
    negative_filename = os.path.join(voc_database, 'ImageSets/Main', 'car_trainval.txt')
    with open(negative_filename, "r") as f:
        for line in f:
            index, flag = line.split()
            if flag != 1:
                counter = random_crop(index, voc_database, autorally_database_path, neg_list, counter)

    # Generating train and test sets
    train_files = os.path.join(autorally_database_path, 'ImageSets/Main', 'train.txt')
    test_files = os.path.join(autorally_database_path, 'ImageSets/Main', 'test.txt')
    # random.shuffle(pos_list)
    # random.shuffle(neg_list)
    n_pos_train = int(0.75*len(pos_list))
    n_neg_train = int(0.75*len(neg_list))
    pos_train = pos_list[0:n_pos_train]
    pos_test = pos_list[n_pos_train:]
    neg_train = neg_list[0:n_neg_train]
    neg_test = neg_list[n_neg_train:]
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




