# -*- coding: utf-8 -*-
import cv2
import json
import os
from sklearn import svm
from sklearn import metrics
import numpy as np
from item import Item
import matplotlib.pyplot as plt

DEBUG = True
DEBUG_TRAINING_IMG = False
DEBUG_TARGET_IMG = False
KEY_TRAINING_PATH = 'training_path'
KEY_TARGET_PATH = 'target_path'
VALID_IMG_TYPE = ('.jpg', '.jpeg', '.png')
NORMALIZED_SIZE = (128, 128)


def main():
    config_json = load_json_file('config.json')
    if not is_valid_config_file(config_json):
        return

    print '[main] Config:', config_json

    training_items = load_images(config_json[KEY_TRAINING_PATH])
    target_items = load_images(config_json[KEY_TARGET_PATH])
    if len(training_items) == 0 or len(target_items) == 0:
        print '[main] No images in training path!'
        return

    if DEBUG:
        print '[main] training items ----------------------------'
        for item in training_items:
            print item
        print '[main] target items ----------------------------'
        for item in target_items:
            print item

    print '[main]', len(training_items), 'training images, ', len(target_items), 'target images are read'
    resize_images(training_items)
    print '[main]', len(training_items), 'training images are normalized, size =', NORMALIZED_SIZE
    resize_images(target_items)
    print '[main]', len(target_items), 'target images are normalized, size =', NORMALIZED_SIZE

    if DEBUG_TRAINING_IMG:
        show_images(training_items)
    if DEBUG_TARGET_IMG:
        show_images(target_items)

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)
    training_imgs = [np.array(item.normalized_img).flatten().tolist() for item in training_items]
    training_class_num = [item.class_num for item in training_items]
    classifier.fit(training_imgs, training_class_num)

    # Now predict the value of the digit on the second half:
    target_imgs = [np.array(item.normalized_img).flatten().tolist() for item in target_items]
    target_class_num = [item.class_num for item in target_items]
    predicted = classifier.predict(target_imgs)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(target_class_num, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(target_class_num, predicted))

    # images_and_predictions = list(zip(training_imgs, predicted))
    # for index, (image, prediction) in enumerate(images_and_predictions[:2]):
    #     plt.subplot(1, 2, index+1)
    #     plt.axis('off')
    #     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.title('Prediction: %i' % prediction)
    #
    # plt.show()


def load_json_file(path):
    try:
        with open(path) as data_file:
            # TODO: turn all key into lower case
            return json.load(data_file)
    except ValueError as e:
        print '[load_json_file] error:', e
        return None


def is_valid_config_file(config_file):
    if config_file and config_file[KEY_TRAINING_PATH] and config_file[KEY_TARGET_PATH]:
        return True
    else:
        if not config_file[KEY_TRAINING_PATH]:
            print '[is_valid_config_file] Can\'t find your training img path in config.'
            print '[is_valid_config_file] Please set attribute \'training_path\' for it.'
        elif not config_file[KEY_TARGET_PATH]:
            print '[is_valid_config_file] Can\'t find your target img path in config.'
            print '[is_valid_config_file] Please set attribute \'target_path\' for it.'
        return False


def load_images(img_path):
    items = []
    for file in os.listdir(img_path):
        file_split_text = os.path.splitext(file)
        file_name = file_split_text[0]
        file_ext = file_split_text[1]
        if file_ext.lower() in VALID_IMG_TYPE:
            file_path = os.path.join(img_path, file)
            img = cv2.imread(file_path)
            class_num = int(file_name.split('_')[0])
            item = Item(img, class_num, file, file_path)
            items.append(item)
    return items


def resize_images(items):
    for item in items:
        item.normalized_img = cv2.resize(item.img, NORMALIZED_SIZE)


def show_images(items):
    imgs = [item.normalized_img for item in items]
    for i, f in enumerate(imgs):
        cv2.imshow('training img ' + str(i), f)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
