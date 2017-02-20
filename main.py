# -*- coding: utf-8 -*-
import cv2
import json
import os

DEBUG = False
VALID_IMG_TYPE = ('.jpg', '.jpeg', '.png')
NORMALIZED_SIZE = (128, 128)


def main():
    config_json = load_json_file('config.json')
    if not is_valid_config_file(config_json):
        return

    print '[main] Config:', config_json
    training_imgs = load_images(config_json['training_path'])
    if len(training_imgs) == 0:
        print '[main] No images in training path!'
        return

    print '[main]', len(training_imgs), 'training images are read'
    normalized_training_imgs = resize_images(training_imgs)
    print '[main]', len(training_imgs), 'training images are normalized'

    if DEBUG:
        for i, f in enumerate(normalized_training_imgs):
            cv2.imshow('training'+str(i), f)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def load_json_file(path):
    try:
        with open(path) as data_file:
            # TODO: turn all key into lower case
            return json.load(data_file)
    except ValueError as e:
        print '[load_json_file] error:', e
        return None


def is_valid_config_file(config_file):
    if config_file and config_file['training_path'] and config_file['target_path']:
        return True
    else:
        print '[is_valid_config_file] Can\'t find your img path in config.'
        print '[is_valid_config_file] Please set attribute \'path\' for it.'
        return False


def load_images(img_path):
    imgs = []
    for f in os.listdir(img_path):
        if os.path.splitext(f)[1].lower() in VALID_IMG_TYPE:
            imgs.append(cv2.imread(os.path.join(img_path, f)))
    return imgs


def resize_images(imgs):
    return [cv2.resize(img, NORMALIZED_SIZE) for img in imgs]


if __name__ == '__main__':
    main()
